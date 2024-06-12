import random
import gym
import numpy as np
import pygame

env = gym.make("CartPole-v1")
obs = env.reset()


def policy(obs):
    angle = obs[2]
    if angle<0:
        return 0
    else:
        return 1


totals=[]


for episode in range(300):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards +=reward

        if done:
            break
        else:
            #Render almamak icin bura iptal
            env.render()
    totals.append(episode_rewards)

print(np.mean(totals),np.std(totals),np.min(totals),np.max(totals))

env.close()


