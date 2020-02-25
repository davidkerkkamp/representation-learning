import os
import time
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

class Object(object):
    pass


# Mock command line arguments
args = Object()
args.method = 'infonce-stdim'
args.feature_size = 256
args.no_downsample = True
args.end_with_relu = False
args.env_name = 'PitfallNoFrameskip-v4'
args.pretraining_steps = 10000
args.cuda_id = ''
args.epochs = 1
args.batch_size = 64
args.patience = 15
args.lr = 3e-4
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock wandb object
wandb = Object()
wandb.run = Object()
wandb.run.dir = 'wandb'
wandb.log = lambda a, step, commit: None

observation_shape = torch.Size([1, 210, 160])
model_dir = 'model-files/'
encoder_model_path = model_dir + args.env_name + '-encoder.pt'
probe_model_path = model_dir + args.env_name + '-{}.pt'


def train_encoder(encoder):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    tr_episodes, val_episodes, \
    tr_labels, val_labels, \
    test_episodes, test_labels = get_episodes(env_name=args.env_name, steps=args.pretraining_steps)
    print('Obtained episodes for training encoder')

    torch.set_num_threads(1)
    config = {}
    config.update(vars(args))
    config['obs_space'] = encoder.input_channels

    print('Training encoder...')
    trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    trainer.train(tr_episodes, val_episodes)
    print('Done training encoder')
    return encoder


def load_encoder():
    encoder = NatureCNN(observation_shape[0], args)  # TODO observation shape
    if os.path.isfile(encoder_model_path):
        print('Encoder model exists, loading weights')
        encoder.load_state_dict(torch.load(encoder_model_path))
        encoder.eval()
    else:
        print(f'No encoder model with name {encoder_model_path} found, training a new encoder')
        encoder = train_encoder(encoder)
        torch.save(encoder.state_dict(), encoder_model_path)
    return encoder


def set_probes(probe_trainer: ProbeTrainer, probes: dict, labels):
    probe_trainer.probes = probes

    probe_trainer.early_stoppers = {
        k: EarlyStopping(patience=probe_trainer.patience, verbose=False, name=k + "_probe",
                         save_dir=probe_trainer.save_dir)
        for k in labels.keys()}

    probe_trainer.optimizers = {k: torch.optim.Adam(list(probe_trainer.probes[k].parameters()),
                                                    eps=1e-5, lr=probe_trainer.lr) for k in labels.keys()}
    probe_trainer.schedulers = {
        k: torch.optim.lr_scheduler.ReduceLROnPlateau(probe_trainer.optimizers[k], patience=5, factor=0.2, verbose=True,
                                                      mode='max', min_lr=1e-5) for k in labels.keys()}


def train_probes(probe_trainer: ProbeTrainer):
    print('Obtaining episodes for probe training')
    tr_episodes, val_episodes, \
    tr_labels, val_labels, \
    test_episodes, test_labels = get_episodes(env_name=args.env_name, steps=args.pretraining_steps)
    print('Training probes')
    probe_trainer.train(tr_episodes, val_episodes, tr_labels, val_labels)
    print('Probe training complete')

    for i, k in enumerate(probe_trainer.probes):
        torch.save(probe_trainer.probes[k].state_dict(), probe_model_path.format(k))
    print(f'Saved {len(probe_trainer.probes)} probe models to {model_dir} directory')


def load_probes(encoder, labels):
    probe_trainer = ProbeTrainer(encoder=encoder, representation_len=encoder.feature_size, epochs=args.epochs)
    probes = dict()
    for i, k in enumerate(labels):
        path = probe_model_path.format(k)
        if not os.path.isfile(path):
            print(f'Probe model for label {k} not found')
            break
        probes[k] = LinearProbe(input_dim=probe_trainer.feature_size,
                                num_classes=probe_trainer.num_classes).to(probe_trainer.device)
        print(f'- Loading state dict for label {k}')
        probes[k].load_state_dict(torch.load(path))
    if len(probes) == len(labels):
        set_probes(probe_trainer, probes, labels)
    else:
        print('Training new probe models')
        train_probes(probe_trainer)
    return probe_trainer


def predict(pt, frame):
    with torch.no_grad():
        pt.encoder.to(args.device)
        f = pt.encoder(frame).detach()
    probes = pt.probes
    preds = dict()
    for i, k in enumerate(probes):
        probes[k].to(args.device)
        p = probes[k](f)
        preds[k] = np.argmax(p.cpu().detach().numpy(), axis=1)[0]
    return preds


def probetrainer_setup(env):
    labels = env.labels()
    encoder = load_encoder()
    probe_trainer = load_probes(encoder, labels)
    return probe_trainer


env = AtariARIWrapper(gym.make('PitfallNoFrameskip-v4'))
env = GrayscaleWrapper(env)
pt = probetrainer_setup(env)
obs = env.reset()
obs, reward, done, info = env.step(0)
# for t in range(200):
#     env.render()
#     obs, reward, done, info = env.step(3)
#     time.sleep(0.05)


obs = obs.reshape(1, 1, 210, 160)
p_obs = torch.from_numpy(obs).float()
prediction = predict(pt, p_obs)
