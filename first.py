# Position code : /home/floflo/anaconda3/lib/python3.7/site-packages/gym/envs/box2d


##############################################################
####### Basic code only doing random action on the car #######
##############################################################


import gym
from gym.envs.box2d import CarRacing
import numpy as np

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from DQN import DQN
from memory import ReplayMemory
from treat_screen import transform_obs
from not_ML import next_action

env = gym.make('CarRacing-v0')
actions = [[0 , 0.3, 0], [-1 , 0.3, 0], [1 , 0.3, 0]]
nb_actions = len(actions)

################ Parameters to change ###############
nb_episodes = 1
size_episode = 50
size_memory = 1000
#####################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

########################################################################
############################  LOOP #####################################
########################################################################

for i_episode in range(nb_episodes):
    observation = env.reset()
    reward_tot = 0
    for t in range(size_episode):
        env.render()
        # action = env.action_space.sample()      # Take a random action
        if (next_action(observation) =='left'):
            action = actions[1]
        else:
            action = actions[2]
        observation, reward, done, info = env.step(action)
        # print('aaaa')
        # find_diff_pixels(observation)
        # reward_tot += reward
        # if (t%10==0):
        #     print(reward, reward_tot)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

########################################################################
############################ END LOOP ##################################
########################################################################

# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


simple_screen = transform_obs(observation[:84])
torch_screen = torch.from_numpy(simple_screen)
screen_height, screen_width = torch_screen.shape
# aze = resize(torch_screen).unsqueeze(0).to(device)
print(env.render(mode='rgb_array').transpose((2, 0, 1)).shape)
torch_screen = torch_screen.unsqueeze(0)
torch_screen = torch_screen.unsqueeze(0)

env.close()

policy_net = DQN(screen_height, screen_width, nb_actions).to(device)
target_net = DQN(screen_height, screen_width, nb_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(size_memory)


print(random.random())

print(policy_net(torch_screen))

