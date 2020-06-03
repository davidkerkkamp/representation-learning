import torch
from AtariARIHandler import AtariARIHandler
from MDPBuilder import MDPBuilder

'''
This file shows how to use the code in this project. By default for Pong, but it works for every of the 22 games 
supported by https://github.com/mila-iqia/atari-representation-learning.
'''


class Object(object):
    pass


# Mock command line arguments needed for AtariARI
args = Object()
args.method = 'infonce-stdim'
args.feature_size = 256
args.no_downsample = True
args.end_with_relu = False
args.env_name = 'PongNoFrameskip-v4'  # Atari env name of the game to use
args.pretraining_steps = 100000  # Steps to use for training encoder
args.probe_steps = 50000  # Steps to use for training linear probes
args.cuda_id = '0'
args.epochs = 100  # Number of training epochs
args.batch_size = 64
args.patience = 15
args.lr = 3e-4
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock wandb object needed for AtariARI
wandb = Object()
wandb.run = Object()
wandb.run.dir = 'wandb'
wandb.log = lambda a, step, commit: None

# Labels to ignore when training/using probes. For labels that are ignored, no probes are trained or loaded.
ignore_labels = ['player_x', 'enemy_x']
handler = AtariARIHandler(args, wandb)
gym_env = handler.get_gym_env()
handler.probe_setup(ignore_labels)  # Train encoder/probe models, or load them if exist

labels_to_use = ['ball_x', 'ball_y', 'player_y']  # Labels to use for MDP, other labels are not collected

# Actions to use for MDP, for Pong only 3 actions are relevant: none (0), up (2) and down (5)
# These correspond to actions defined by Arcade Learning Environment
actions_to_use = [0, 2, 5]

mdp_builder = MDPBuilder(labels_to_use, actions_to_use, log=False)
obs = gym_env.reset()  # Reset env before use
n_steps = 1000  # No. of steps to run on the Gym environment
for i in range(n_steps):
    # gym_env.render()  # Optionally render the game to the screen
    action = mdp_builder.get_random_action()  # Get an action to apply on the env
    obs, reward, done, info = gym_env.step(action)  # Perform step on env using given action
    prediction = handler.predict(obs)  # Obtain prediction using observation from env
    mdp_builder.add_state_info(prediction, action)  # Add the predicted info to the MDPBuilder

    # Instead of prediction, MDP can be built using ground truth available in info['labels']
    # mdp_builder.add_state_info(info['labels'], action)

    if done:  # Game finished, reset env to continue
        print(f'Resetting env (step {i})')
        gym_env.reset()
        mdp_builder.restart()  # Treat next observed state repr as initial state
print(f'DONE: found {mdp_builder.num_states()} states')

# - Save the MDPBuilder state to a file:
# mdp_builder.save_builder_to_file('mdp_builder/mdp_1.pkl')
# - Can be loaded to continue building later using code below:
# mdp_builder = MDPBuilder(labels_to_use, actions_to_use)
# mdp_builder.load_from_file('mdp_builder/mdp_1.pkl')

# - Export MDP in given format
# mdp_builder.build_model_file('mdp.pm', format='prism')
