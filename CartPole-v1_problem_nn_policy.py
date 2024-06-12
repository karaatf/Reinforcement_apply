import tensorflow as tf
from tensorflow import keras
from keras import layers
import gym
import numpy as np
import pygame

env = gym.make("CartPole-v1")
obs = env.reset()

n_inputs = 4


model = keras.models.Sequential()
model.add(layers.Dense(5,activation="relu",input_shape=n_inputs))
model.add(layers.Dense(1,activation="sigmoid"))


def play_one_step(env, obs, model, loss_func):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1.])>left_proba)
        y_target = tf.constant([[1.]])-tf.cast(action,tf.float32)
        loss = tf.reduce_mean(loss_func(y_target,left_proba))
    grads = tape.gradient(loss,model.trainable_variables)
    obs,reward,done,info = env.step(int(action[0,0].numpy()))

    return obs,reward,done,info


def play_multiple_episodes(env,n_episodes,n_max_steps,model,loss_func):
    all_rewards=[]
    all_grads=[]
    for episode in range(n_episodes):
        current_rewards=[]
        current_grads=[]
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads= play_one_step(env,obs,model,loss_func)
            current_rewards.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards,all_grads


def discount_rewards(rewards,dicount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards)):
        discounted[step]+=discounted[step+1]*dicount_factor
    return  discounted


def discount_and_normalize_rewards(all_rewards,discount_factor):
    all_discounted_rewards=[discount_rewards(rewards,discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std=flat_rewards.std()
    return [(discounted_rewards-reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

n_iteration=150
n_episodes = 10
n_max_step=200
discount_factor=0.95

optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy()
