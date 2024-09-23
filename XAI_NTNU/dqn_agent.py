import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from IPython import display
import wandb

from DQNGridWorldEnv import DQNGridWorldEnv

#env = DQNGridWorldEnv(render_mode=None, size=4, goalReward=100, stepLoss=-1)

#env = gym.make('CartPole-v1')
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
#    from IPython import display
# plt.ion()

class neural_network(torch.nn.Module):
    def __init__(self, observation_space_n, action_space_n):
        super(neural_network, self).__init__()
        self.layer1 = torch.nn.Linear(observation_space_n, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, action_space_n)

    def forward(self, x):
        x = torch.nn.Flatten()(x)
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
    def __init__(self, action_space, observation, batch_size=128, lr=0.0001, gamma=0.99, epsilon_start=0.9, epsilon_min=0.05, epsilon_decay=1000, tau=0.005, wandb=None):
        self.action_space = action_space
        self.observation = observation
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.wandb = wandb

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")
        
        self.policy_network = neural_network(np.prod(self.observation.shape), self.action_space.n).to(self.device)
        self.target_network = neural_network(np.prod(self.observation.shape), self.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_rewards = []
        self.epsilon_history = []
        self.loss = 0.0
        self.loss_history = []

    def get_action(self, state):
        sample = random.random()
        # Epsilon-greedy action selection with exponential decay
        epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if epsilon > sample:
            return torch.tensor([[random.randrange(self.action_space.n)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)

    def plot_rewards(self, show_result=False):
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        epsilons_t = torch.tensor(self.epsilon_history, dtype=torch.float)
        loss_t = torch.tensor(self.loss_history, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf() # This clears the figure
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.plot(rewards_t.numpy())
        plt.plot(epsilons_t.numpy())
        plt.plot(loss_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        
        plt.pause(0.001)  # pause a bit so that plots are updated
        if not show_result:
            # display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf()) # Comment out to stop printing?

    def optimize_model(self):
        # The model doesn't start training before having sufficient data
        if len(self.memory) < self.batch_size:
            return
        # A transition is a tuple of (state, action, next_state, reward)
        # , basically an experience for the agent that matches a
        # state + action with a new state + reward
        # These are sampled randomly to improve stability for the training
        transitions = self.memory.sample(self.batch_size)
        
        # Transposes the batch-array of Transitions
        # to Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        # Creates a mask of all non-final states
        # to distinguish between state where further
        # actions are possible and those where the game has ended
        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None])
        
        # torch.cat concatenates the input tensor along the specified dimension
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(
                non_final_next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        self.loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Unsqueezing is used to add an extra dimension to the tensor
        
        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

    def train(self, env, num_episodes=200):
        # Loop through episodes
        for i_episode in range(num_episodes):
            # Init env and git its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_reward = 0
            for i in count(): # Loop through steps
                action = self.get_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32,
                        device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # print(f"Episode: {i_episode}, Step: {i}, episode reward: {episode_reward}, truncated: {truncated}, terminated: {terminated}")

                # Move to the next state
                state = next_state

                # Perform one step of the optimization
                self.optimize_model()

                # Soft update the target network's weights
                target_net_state_dict = self.target_network.state_dict()
                policy_net_state_dict = self.policy_network.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_network.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_rewards.append(episode_reward)
                    epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
                    self.epsilon_history.append(epsilon)
                    self.loss_history.append(self.loss)
                    self.wandb.log({"episode_reward": episode_reward,
                                    "epsilon": epsilon,
                                    "loss": self.loss})
                    self.plot_rewards()
                    break
            
        print("Complete")
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()

    def save_model(self):
        pass

    def load_model(self):
        pass

# main
if __name__ == "__main__":

    # Config
    num_episodes = 10_000

    # DQNGridWorldEnv
    size=8
    goalReward=2
    stepLoss=-0.01
    maxSteps=100
    wallCoordinates=None

    # Agent
    batch_size=128
    lr=0.0001
    gamma=0.99
    epsilon_start=0.9
    epsilon_min=0.05
    epsilon_decay=50_000
    tau=0.005

    wandb.init(project=f"{size}x{size}_{num_episodes}episodes",
        config={
        "size": size,
        "goalReward": goalReward,
        "stepLoss": stepLoss,
        "maxSteps": maxSteps,
        "batch_size": batch_size,
        "lr": lr,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "tau": tau
        })
    env = DQNGridWorldEnv(render_mode=None, size=size, goalReward=goalReward, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates)
    observation, info = env.reset()
    agent = dqn_agent(env.action_space, observation,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        tau=tau,
        wandb=wandb)
    agent.train(env=env, num_episodes=num_episodes)

    #show_env = DQNGridWorldEnv(render_mode="human", size=size, goalReward=goalReward, stepLoss=-stepLoss, maxSteps=maxSteps)
    #agent.train(env=show_env, num_episodes=50)