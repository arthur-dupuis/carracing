
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

size_episode = 300

env = gym.make('CarRacing-v0')

actions = [[0 , 0.3, 0], [-1 , 0.3, 0], [1 , 0.3, 0]]
nb_actions = len(actions)
screen_height, screen_width = 84, 96


policy_net = DQN(screen_height, screen_width, nb_actions)
policy_net.load_state_dict(torch.load('./project/carracing/models/model'))
policy_net.eval()


########################################################################
############################  LOOP #####################################
########################################################################
tot_reward = 0
observation = env.reset()
for t in range(size_episode):

    state = transform_obs(observation)
    action_index = policy_net(state).max(1)[1].view(1, 1).item()
    action = actions[action_index]
    observation, reward, done, info = env.step(action)

    env.render()
    tot_reward += reward

    # # action = env.action_space.sample()      # Take a random action
    # if (next_action(observation) =='left'):
    #     action = actions[1]
    # else:
    #     action = actions[2]
    # observation, reward, done, info = env.step(action)
    # # find_diff_pixels(observation)
    # # reward_tot += reward
    # # if (t%10==0):
    # #     print(reward, reward_tot)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
print(tot_reward)
########################################################################
############################ END LOOP ##################################
########################################################################