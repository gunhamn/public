from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

from gridWorldEnv import GridWorldEnv
from qLearningAgent import qLearningAgent


env = GridWorldEnv(render_mode=None, size=5)
done = False
observation, info = env.reset()


# hyperparameters
learning_rate = 0.01
n_episodes = 10_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = qLearningAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    env=env,
)



for game in range(5000): # 100 games
    print(f"Game {game}")
    if game % 500 == 499:
        env = GridWorldEnv(render_mode="human", size=5)
        observation, info = env.reset()
        print(f"q_values: {agent.q_values}")
    else:
        env = GridWorldEnv(render_mode=None, size=5)
        observation, info = env.reset()
    for step in range(50): # max 50 steps
        observation = env._get_obs()
        action = agent.get_action(observation)  # agent policy that uses the observation and info
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.update(observation, action, reward, terminated, next_observation)
        agent.decay_epsilon()

        if terminated or truncated:
            break # start a new game

env.close()
