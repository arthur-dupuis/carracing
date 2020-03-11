import gym
from DeepQLearning import DeepQLearning as DQL

env = gym.make("CarRacing-v0")

trainings = 1000  # number of training episodes
saving_frequence = 400  # frequence of saving checkpoints

config = dict()

agent = DQL(config)


def save():
    return


def run_once():
    reward, frames = agent.run()
    if agent.count % saving_frequence == 0:
        save()


def main():
    while True:
        if agent.count > trainings:
            break
        run_once()
