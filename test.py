
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

size_episode = 2000

env = gym.make('CarRacing-v0')

actions = [[0 , 0.3, 0.1], [1 , 0.3, 0.1], [-1 , 0.3, 0.1]]
nb_actions = len(actions)
screen_height, screen_width = 84, 96


policy_net = DQN(screen_height, screen_width, nb_actions)
policy_net.load_state_dict(torch.load('./models/bestmodel_saved'))
policy_net.eval()


########################################################################
############################  LOOP #####################################
########################################################################
tot_reward = 0
observation = env.reset()
for t in range(size_episode):

    on_grass , state = transform_obs(observation)
    # At first, the data displayed is not relevant for the data on which the model is trained
    if t<50:
        action = actions[0]
    else:
        action_index = policy_net(state).max(1)[1].view(1, 1).item()
        action = actions[action_index]

    observation, reward, done, info = env.step(action)
    env.render()
    if(reward<0):
        if(on_grass and t> 50):
            reward = float(-1)
        if(not on_grass and t>50):
            reward = float(0.1)
    tot_reward += reward

    # if done:
    #     print("Episode finished after {} timesteps".format(t+1))
    #     break
print(tot_reward)
########################################################################
############################ END LOOP ##################################
########################################################################