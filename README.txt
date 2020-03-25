Project INF581 - Car Racing : Arthur Dupuis - Florent Ozouf

The folder is structured as below :

/Carracing
    |
    |--> train.py
    |--> test.py
    |--> treat_screen.py
    |--> DQN.py
    |--> memory.py
    |
    |--> /models
           |
           |--> model
           |--> bestmodel
           |--> bestmodel_saved

1) The file train.py :

This file is the file to lauch to train a model.
This file declare all the instances.
It countains many parameters of the model and of the loop.
It countain alors the following function : select_action, optimize_model.
It countains the training loop.

2) The file test.py

This file allows us to se the performances of a model.
We can choose which model to test. 
By default, it is the bestmodel_saved

3) The file treate_screen.py :

This file countains 2 function : transform_obs and print_screen :
The function tranform_obs transform and observation of 96x96x3 array in a 96x84 boolean array telling us if the corresponding pixel countains grass or not. It also tells us if the agent is on the grass or not.
The function print_screen only print(the screen obtained after transformation). It was a usefull function to debug our code.

4) The file DQN.py :

This file countains the class DQN.
This call defines and creates neural networks and defines the function forward.

5) The file memory.py :

This file defines the class ReplayMemory usefull for a DQN model.

6) The folder model.

We stocked our models in this folder.


