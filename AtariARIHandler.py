import os
import gym
import torch
import numpy as np
from atariari.benchmark.episodes import get_episodes
from atariari.benchmark.probe import ProbeTrainer, LinearProbe
from atariari.benchmark.utils import EarlyStopping
from atariari.benchmark.wrapper import AtariARIWrapper
from atariari.methods.encoders import NatureCNN
from atariari.methods.stdim import InfoNCESpatioTemporalTrainer
from atariari.benchmark.envs import GrayscaleWrapper


# Based on https://github.com/mila-iqia/atari-representation-learning


class AtariARIHandler:
    def __init__(self, args, wandb,
                 observation_shape=torch.Size([1, 210, 160]),
                 model_dir='model-files'):
        self.args = args  # object containing all needed parameters for AtariARI to work
        self.wandb = wandb  # wandb is not used in this project, but still a (mock) object is needed
        self.observation_shape = observation_shape  # observation tensor shape of Atari observations
        self.model_dir = model_dir + ('' if model_dir[-1] == '/' else '/')  # dir where encoder/probe models are stored
        self.encoder_model_path = self.model_dir + args.env_name + '-encoder.pt'  # file name of encoder model
        self.probe_model_path = self.model_dir + args.env_name + '-{}.pt'  # file name format for probe models
        self.probe_trainer = None  # object of type ProbeTrainer
        gym_env = AtariARIWrapper(gym.make(self.args.env_name))  # Create Atari env
        self.gym_env = GrayscaleWrapper(gym_env)

    def get_gym_env(self):
        return self.gym_env

    def train_encoder(self, encoder):
        device = torch.device("cuda:" + str(self.args.cuda_id) if torch.cuda.is_available() else "cpu")
        tr_episodes, val_episodes, \
        tr_labels, val_labels, \
        test_episodes, test_labels = get_episodes(env_name=self.args.env_name, steps=self.args.pretraining_steps,
                                                  train_mode="train_encoder")
        print('Obtained episodes for training encoder')

        torch.set_num_threads(1)
        config = {}
        config.update(vars(self.args))
        config['obs_space'] = encoder.input_channels

        print('Training encoder...')
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=self.wandb)
        trainer.train(tr_episodes, val_episodes)
        print('Done training encoder')
        return encoder

    def load_encoder(self):
        encoder = NatureCNN(self.observation_shape[0], self.args)
        if os.path.isfile(self.encoder_model_path):
            print('Encoder model exists, loading weights')
            encoder.load_state_dict(torch.load(self.encoder_model_path, map_location=self.args.device))
            encoder.eval()
        else:
            print(f'No encoder model with name {self.encoder_model_path} found, training a new encoder')
            encoder = self.train_encoder(encoder)
            torch.save(encoder.state_dict(), self.encoder_model_path)
        return encoder

    def set_probes(self, probe_trainer: ProbeTrainer, probes: dict, labels):
        probe_trainer.probes = probes

        probe_trainer.early_stoppers = {
            k: EarlyStopping(patience=probe_trainer.patience, verbose=False, name=k + "_probe",
                             save_dir=probe_trainer.save_dir)
            for k in labels.keys()}

        probe_trainer.optimizers = {k: torch.optim.Adam(list(probe_trainer.probes[k].parameters()),
                                                        eps=1e-5, lr=probe_trainer.lr) for k in labels.keys()}
        probe_trainer.schedulers = {
            k: torch.optim.lr_scheduler.ReduceLROnPlateau(probe_trainer.optimizers[k], patience=5, factor=0.2,
                                                          verbose=True,
                                                          mode='max', min_lr=1e-5) for k in labels.keys()}

    def train_probes(self, probe_trainer: ProbeTrainer):
        print('Obtaining episodes for probe training')
        tr_episodes, val_episodes, \
        tr_labels, val_labels, \
        test_episodes, test_labels = get_episodes(env_name=self.args.env_name, steps=self.args.probe_steps)
        print('Training probes')
        probe_trainer.train(tr_episodes, val_episodes, tr_labels, val_labels)
        print('Probe training complete')

        for i, k in enumerate(probe_trainer.probes):
            torch.save(probe_trainer.probes[k].state_dict(), self.probe_model_path.format(k))
        print(f'Saved {len(probe_trainer.probes)} probe models to {self.model_dir} directory')

    def load_probes(self, encoder, labels):
        probe_trainer = ProbeTrainer(encoder=encoder, representation_len=encoder.feature_size, epochs=self.args.epochs)
        probes = {}
        for i, k in enumerate(labels):
            path = self.probe_model_path.format(k)
            if not os.path.isfile(path):
                print(f'Probe model for label {k} not found')
                break
            probes[k] = LinearProbe(input_dim=probe_trainer.feature_size,
                                    num_classes=probe_trainer.num_classes).to(probe_trainer.device)
            print(f'- Loading probe model for label {k}')
            probes[k].load_state_dict(torch.load(path, map_location=self.args.device))
        if len(probes) == len(labels):
            self.set_probes(probe_trainer, probes, labels)
        else:
            print('Training new probe models')
            self.train_probes(probe_trainer)
        return probe_trainer

    def probe_setup(self, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = []
        labels = self.gym_env.labels()
        for lbl in ignore_labels:
            try:
                del labels[lbl]
                print(f'Ignoring label {lbl}')
            except:
                print(f'Ignore label {lbl}: no such label')

        encoder = self.load_encoder()
        self.probe_trainer = self.load_probes(encoder, labels)
        print('Encoder en probe setup complete')

    def predict(self, obs):
        assert self.probe_trainer is not None, 'ProbeTrainer is not initialized, call probetrainer_setup() first'
        pt = self.probe_trainer
        obs = obs.reshape(1, 1, 210, 160)
        obs = torch.from_numpy(obs).float()
        with torch.no_grad():
            pt.encoder.to(self.args.device)
            obs = obs.to(self.args.device)
            f = pt.encoder(obs).detach()
        probes = pt.probes
        preds = {}
        for i, k in enumerate(probes):
            probes[k].to(self.args.device)
            p = probes[k](f)
            preds[k] = np.argmax(p.cpu().detach().numpy(), axis=1)[0]
        return preds
