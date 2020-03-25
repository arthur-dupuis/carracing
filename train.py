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
from treat_screen import transform_obs, print_screen

env = gym.make('CarRacing-v0')


################ Action state ######################
actions = [[0 , 0.3, 0.08], [1 , 0.3, 0.1], [-1 , 0.3, 0.1]]
nb_actions = len(actions)
#####################################################

################ Parameters to change ###############
nb_episodes = 100
basic_size_episode = 300
size_memory = 100000
#####################################################

################ Network Parameters  ###############
BATCH_SIZE = 100
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
screen_height, screen_width = 84, 96

policy_net = DQN(screen_height, screen_width, nb_actions).to(device)
# policy_net.load_state_dict(torch.load('./models/model'))
target_net = DQN(screen_height, screen_width, nb_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(size_memory)
##########################################################


################### Select Action #########################
def select_action(state):
    global steps_done
    sample = random.random() # Between 0 and 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold: #Action determined by the NN
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else: # Random Action
        return torch.tensor([[random.randrange(nb_actions)]], device=device, dtype=torch.long)
##########################################################


########################################################################
############################ OPTIMIZE ##################################
########################################################################
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)# Transpose the batch
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.stack(tuple(torch.from_numpy(numpy.array(batch.action))))
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.long().unsqueeze(1))
    # state_action_values = state_action_values1.gather(1, action_batch.long())

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
best_reward = -500
size_episode = basic_size_episode


########################################################################
############################  LOOP #####################################
########################################################################

for i_episode in range(nb_episodes):    # Initialize the environment and state
    observation = env.reset()
    _, last_screen = transform_obs(observation)
    _, current_screen = transform_obs(observation)
    state = current_screen
    tot_reward = 0
    for t in range(size_episode):
        # env.render()
        action_index = select_action(state).item()
        if(t<=50):           # Before t = 50, the zoom of the screen displayed is not permanent,
            action_index = 0 # so the data isn't relevant to learn
        observation, reward, done, info = env.step(actions[action_index])
        last_screen = current_screen
        on_grass, current_screen = transform_obs(observation)
        # Change of the reward to add penalty when the agent isn't on the road
        if(reward<0):
            if(on_grass and t> 50):
                reward = float(-1)
            if(not on_grass and t>50):
                reward = float(0.1)
        if(t<=50):
            reward = float(0)
        reward = torch.tensor([reward], device=device)


        # Store the transition in memory
        memory.push(last_screen, action_index, current_screen, reward)

        # Move to the next state
        state = current_screen

        # Perform one step of the optimization (on the target network)
        optimize_model()
        tot_reward += reward
        if done:
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), './models/model') # Save the model

    if(tot_reward > 0): # The more the agent is well trained, the more the episode is long
        size_episode = 300 + int(tot_reward)
    else:
        size_episode = basic_size_episode

    print(tot_reward)
    
    if(tot_reward > best_reward): # If it becomes the best model : save it 
        torch.save(policy_net.state_dict(), './models/bestmodel')
        best_reward = tot_reward
        print('New Best')

########################################################################
############################ END LOOP ##################################
########################################################################

env.close()

