from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

class qLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        env: gym.Env = None,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.training_error = []

    def get_action(self, observation: dict) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.

        Args:
            observation: A dictionary with 'agent' and 'target' positions.
        """
        # Convert the observation dictionary to a tuple that can be used as a key in q_values
        observation = tuple(map(tuple, observation.values()))

        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[observation]))

    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        terminated: bool,
        next_observation: dict,
    ):
        # Convert the observation dictionaries to tuples of tuples
        observation = tuple(map(tuple, observation.values()))
        next_observation = tuple(map(tuple, next_observation.values()))

        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_observation])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[observation][action])

        self.q_values[observation][action] =(
            self.q_values[observation][action] + self.learning_rate * temporal_difference)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)