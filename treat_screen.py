import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def transform_obs(obs):
    dat = np.zeros((len(obs), len(obs[0])), dtype=int)
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if (obs[i][j][1] > obs[i][j][0] and obs[i][j][1] > obs[i][j][2]):
                dat[i][j] = int(0)
            else :
                dat[i][j] = int(1)
    simple_screen = dat[:84]
    torch_screen = torch.from_numpy(simple_screen)
    torch_screen = torch_screen.unsqueeze(0)
    torch_screen = torch_screen.unsqueeze(0).float()
    return torch_screen