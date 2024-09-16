from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

from DQNGridWorldEnv import DQNGridWorldEnv
from DQNAgent import DQNAgent

# Interaction hyperparameters
n_episodes = 500
#render_after = 500

# Agent hyperparameters
buffer_size=10000
batch_size=64
gamma=0.95
lr=0.001
epsilon_start=1.0
epsilon_min=0.01
epsilon_decay=0.995

# Environment hyperparameters
size = 6
goalReward = 100
stepLoss = -1
episodeLength = 100

env = DQNGridWorldEnv(render_mode=None, size=size, goalReward=goalReward, stepLoss=stepLoss)

agent = DQNAgent(
    env=env,
    buffer_size=buffer_size,
    batch_size=batch_size,
    gamma=gamma,
    lr=lr,
    epsilon_start=epsilon_start,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay
)

if __name__ == "__main__":
    env = env
    env_rendered = DQNGridWorldEnv(render_mode="human", size=size, goalReward=goalReward, stepLoss=stepLoss)
    print(f"Obsevation space: {env.observation_space}")
    print(f"Obsevation space.shape: {env.observation_space.shape}")
    print(f"Obsevation space.shape[0]: {env.observation_space.shape[0]}")
    
    agent = DQNAgent(env)
    agent.train(episodes=n_episodes, ep_length=episodeLength, plot=True)
    agent.train(1, env=env_rendered, ep_length=episodeLength, printChoices=True)
    agent.train(episodes=n_episodes, ep_length=episodeLength, plot=True)
    agent.train(1, env=env_rendered, ep_length=episodeLength, printChoices=True)
    env.close()
    """
    print(f"obs: {env._get_obs()}")
    agent.train(1, env=env_rendered, ep_length=episodeLength, printChoices=True)
    print(f"Q_values: \n{agent.getQValues(env=env_rendered)}")
    print(f"bestDirection: \n{agent.getBestDirection(env=env_rendered)}")
    print(f"Q_values: \n{agent.getQValues(env=env_rendered)}")
    print(f"bestDirection: \n{agent.getBestDirection(env=env_rendered)}")
    print(f"_target_location: \n{env_rendered._target_location}")
    agent.train(100, ep_length=episodeLength)
    agent.train(1, env=env_rendered, ep_length=episodeLength, printChoices=True)
    print(f"Q_values: \n{agent.getQValues(env=env_rendered)}")
    print(f"bestDirection: \n{agent.getBestDirection(env=env_rendered)}")
    print(f"_target_location: \n{env_rendered._target_location}")
    agent.train(100, ep_length=episodeLength)
    agent.train(1, env=env_rendered, ep_length=episodeLength, printChoices=True)
    print(f"Q_values: \n{agent.getQValues(env=env_rendered)}")
    print(f"bestDirection: \n{agent.getBestDirection(env=env_rendered)}")
    print(f"_target_location: \n{env_rendered._target_location}")
    agent.train(100, ep_length=episodeLength)
    agent.train(1, env=env_rendered, ep_length=episodeLength, printChoices=True)
    print(f"Q_values: \n{agent.getQValues(env=env_rendered)}")
    print(f"bestDirection: \n{agent.getBestDirection(env=env_rendered)}")
    print(f"_target_location: \n{env_rendered._target_location}")
    agent.train(100, ep_length=episodeLength)
    agent.train(1, env=env_rendered, ep_length=episodeLength, printChoices=True)
    print(f"Q_values: \n{agent.getQValues(env=env_rendered)}")
    print(f"bestDirection: \n{agent.getBestDirection(env=env_rendered)}")
    print(f"_target_location: \n{env_rendered._target_location}")
    agent.save_model('dqn_GridWorldEnv_500ep.pth')
env.close()
"""