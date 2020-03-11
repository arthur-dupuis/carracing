# Position code : /home/floflo/anaconda3/lib/python3.7/site-packages/gym/envs/box2d


##############################################################
####### Basic code only doing random action on the car #######
##############################################################


import gym
from gym.envs.box2d import CarRacing
import numpy as np
di = []

def find_diff_pixels(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if (not_in_arr(arr[i][j])):
                di.append([arr[i][j][0], arr[i][j][1], arr[i][j][2]])

def not_in_arr(el):
    print(di)
    print(len(di))
    for i in range(len(di)):
        if (el[0] == di[i][0] and el[1] == di[i][1] and el[2] == di[i][2]):
            return False
    return True

def transform_obs(obs):
    dat = np.zeros((len(obs), len(obs[0])))
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if (obs[i][j][1] > obs[i][j][0] and obs[i][j][1] > obs[i][j][2]):
                dat[i][j] = -1
            else :
                dat[i][j] = 1
    return dat


def next_action(observation):
    road_data = transform_obs(observation)
    l = 0
    r = 0
    for i in range(len(road_data[66])):
        if (i<48):
            l += road_data[66][i]
        else:
            r += road_data[66][i]
    if l>r :
        return 'left'
    else :
        return 'right'


env = gym.make('CarRacing-v0')

for i_episode in range(1):
    observation = env.reset()

    for t in range(1000):
        env.render()
        # action = env.action_space.sample()      # Take a random action
        action_s = [0 , 0.3, 0.1]
        action_l = [-1 , 0.2, 0.1]
        action_r = [1 , 0.2, 0.1]
        if (next_action(observation) =='left'):
            action = action_l
        else:
            action = action_r
        observation, reward, done, info = env.step(action)
        # print('aaaa')
        # find_diff_pixels(observation)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print(observation[0][48])
print(observation[66][48])
env.close()