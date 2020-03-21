import numpy as np

def transform_obs(obs):
    dat = np.zeros((len(obs), len(obs[0])), dtype=int)
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if (obs[i][j][1] > obs[i][j][0] and obs[i][j][1] > obs[i][j][2]):
                dat[i][j] = int(0)
            else :
                dat[i][j] = int(1)
    return dat