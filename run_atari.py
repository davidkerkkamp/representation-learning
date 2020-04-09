import torch

from AtariARIHandler import AtariARIHandler


class Object(object):
    pass


# Mock command line arguments needed for AtariARI
args = Object()
args.method = 'infonce-stdim'
args.feature_size = 256
args.no_downsample = True
args.end_with_relu = False
args.env_name = 'PongNoFrameskip-v4'  # Atari env name of the game to use
args.pretraining_steps = 10000  # Steps to use for training encoder
args.probe_steps = 10000  # Steps to use for training linear probes
args.cuda_id = '0'
args.epochs = 1  # Number of training epochs
args.batch_size = 64
args.patience = 15
args.lr = 3e-4
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock wandb object needed for AtariARI
wandb = Object()
wandb.run = Object()
wandb.run.dir = 'wandb'
wandb.log = lambda a, step, commit: None

ignore_labels = ['player_x', 'enemy_x']  # Labels to ignore when training/using probes
handler = AtariARIHandler(args, wandb)
gym_env = handler.get_gym_env()
handler.probe_setup(ignore_labels)  # Train encoder/probe models, or load them if exist

obs = gym_env.reset()  # Reset env before use
for i in range(400):
    # gym_env.render()  # Render the game to the screen
    obs, reward, done, info = gym_env.step(gym_env.action_space.sample())  # Perform step on env using given action
    if done:
        print('Resetting env')
        gym_env.reset()
    prediction = handler.predict(obs)
    print(prediction)
