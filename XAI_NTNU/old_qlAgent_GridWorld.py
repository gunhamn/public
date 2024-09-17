from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

from public.XAI_NTNU.old_gridWorldEnv import GridWorldEnv
from public.XAI_NTNU.old_qLearningAgent import qLearningAgent

# hyperparameters
n_episodes = 1000_000
render_after = 15000

learning_rate = 0.01
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

goalReward = 150
stepLoss = -1
episodeLength = 100

size = 7
env = GridWorldEnv(render_mode=None, size=size, goalReward=goalReward, stepLoss=stepLoss)
done = False
observation, info = env.reset()

agent = qLearningAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    env=env,
)



for game in range(n_episodes):
    print(f"Game {game}")
    if game % render_after == render_after-1:
        env = GridWorldEnv(render_mode="human", size=size)
        observation, info = env.reset()
        env.setCellQValues(q_values=agent.q_values, goalPosition=observation['target'], normalize=True)
        print(f"Target position: {observation['target']}")
    else:
        env = GridWorldEnv(render_mode=None, size=size)
        observation, info = env.reset()
    for step in range(episodeLength):
        if game % 5000 == 4999:
            env.setCellQValues(q_values=agent.q_values, goalPosition=observation['target'], normalize=True)
        observation = env._get_obs()
        action = agent.get_action(observation)  # agent policy that uses the observation and info
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.update(observation, action, reward, terminated, next_observation)
        agent.decay_epsilon()

        if terminated or truncated:
            break # start a new game

env.close()
