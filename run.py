import gym
from DeepQLearning import DeepQLearning as DQN

env = gym.make("CarRacing-v0")

trainings = 1000  # number of training episodes

config = dict()

agent = DQN(config)


def run_once():


def main():
    while True:
        if count > trainings:
            break
        run_once()
