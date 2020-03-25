# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T


# def transform_obs(obs):
#     count_grass = 0
#     on_grass = False
#     w,h = len(obs), 84
#     dat = np.zeros((w, h), dtype=int)
#     for i in range(w):
#         for j in range(84):
#             if (obs[i][j][1] > obs[i][j][0] and obs[i][j][1] > obs[i][j][2]):
#                 dat[i][j] = int(0)
#                 count_grass += 1
#             else :
#                 dat[i][j] = int(1)
#     torch_screen = torch.from_numpy(dat)
#     torch_screen = torch_screen.unsqueeze(0)
#     torch_screen = torch_screen.unsqueeze(0).float()
#     if (count_grass/(h*w) > 0.87):
#         on_grass = True
#     else:
#         on_grass = False
#     # if (count_grass/(h*w) > 0.8):
#     #     print(dat)
#     return torch_screen, on_grass


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
    on_grass = False
    # We check if the data just at the left and at the right of the car is green or not
    if(simple_screen[70][46]==0 or simple_screen[70][50]==0): 
        on_grass = True
    return on_grass, torch_screen

# A usefull function to debug sometimes
def print_screen(obs):
    dat = np.zeros((len(obs), len(obs[0])), dtype=int)
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if (obs[i][j][1] > obs[i][j][0] and obs[i][j][1] > obs[i][j][2]):
                dat[i][j] = int(0)
            else :
                dat[i][j] = int(1)
    simple_screen = dat[:84]
    print(simple_screen)
    return