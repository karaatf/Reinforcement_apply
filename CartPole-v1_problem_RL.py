import random
import gym
import pygame

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


env = gym.make("CartPole-v1")

episodes = 10

for episode in range(1,episodes+1):
    state=env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0,1])
        _,reward,done,_ = env.step(action)
        score +=reward
        env.render()

    print(f"Episode: {episode}, Score: {score}")

env.close()

