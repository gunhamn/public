import numpy as np
import random
from collections import namedtuple, deque
import torch
import gymnasium as gym
from old_DQNGridWorldEnv import DQNGridWorldEnv

env = DQNGridWorldEnv(render_mode=None, size=4, goalReward=100, stepLoss=-1)

#env = gym.make('CartPole-v1')

"""
Memory buffer
Neural net
DQN_agent
"""

class neural_network(torch.nn.Module):
    def __init__(self, observation_space_n, action_space_n):
        super(neural_network, self).__init__()
        #input_dim = input_shape[0] * input_shape[1] * input_shape[2]  # Flatten input
        self.layer1 = torch.nn.Linear(observation_space_n, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, action_space_n)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class dqn_agent():
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        pass
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")
        
        print(f"Network input: {(np.prod(self.observation_space.shape), self.action_space.n)}")
        self.policy_network = neural_network(np.prod(self.observation_space.shape), self.action_space.n).to(self.device)
        self.target_network = neural_network(np.prod(self.observation_space.shape), self.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())



    def train(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
