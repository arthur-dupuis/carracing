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
nb_episodes = 50
size_episode = 100
size_memory = 1000
##########################################################

################ Network Parameters  ###############
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
##########################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################ Memory ######################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
##########################################################

################## Declare networks ######################
observation = env.reset()


screen_height, screen_width = 84, 96

policy_net = DQN(screen_height, screen_width, nb_actions).to(device)
target_net = DQN(screen_height, screen_width, nb_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(size_memory)
##########################################################

################### Select State #########################
def select_action(state):
    global steps_done
    sample = random.random() # Between 0 and 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(nb_actions)]], device=device, dtype=torch.long)
##########################################################

########################################################################
############################ OPTIMIZE ##################################
########################################################################
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

########################################################################
######################## END OPTIMIZE ##################################
########################################################################


steps_done = 0

########################################################################
############################  LOOP #####################################
########################################################################

for i_episode in range(nb_episodes):    # Initialize the environment and state
    observation = env.reset()
    last_screen = transform_obs(observation)
    current_screen = transform_obs(observation)
    state = current_screen
    tot_reward = 0
    for t in range(size_episode):
        # Select and perform an action
        action_index = select_action(state).item()
        print(action_index)
        action = actions[action_index]
        observation, reward, done, info = env.step(action)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = transform_obs(observation)
        if not done:
            next_state = current_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(last_screen, action, current_screen, reward)

        # Move to the next state
        state = current_screen

        # Perform one step of the optimization (on the target network)
        optimize_model()
        tot_reward += reward
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(tot_reward)

    # observation = env.reset()
    # reward_tot = 0
    # for t in range(size_episode):
    #     env.render()
    #     # action = env.action_space.sample()      # Take a random action
    #     if (next_action(observation) =='left'):
    #         action = actions[1]
    #     else:
    #         action = actions[2]
    #     observation, reward, done, info = env.step(action)
    #     # print('aaaa')
    #     # find_diff_pixels(observation)
    #     # reward_tot += reward
    #     # if (t%10==0):
    #     #     print(reward, reward_tot)
    #     if done:
    #         print("Episode finished after {} timesteps".format(t+1))
    #         break

########################################################################
############################ END LOOP ##################################
########################################################################

# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])






# print(random.random())

env.close()

