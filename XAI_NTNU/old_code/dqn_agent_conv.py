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
import time
from datetime import datetime
import wandb


from DQNGridWorldEnvConv import DQNGridWorldEnvConv

#env = DQNGridWorldEnv(render_mode=None, size=4, goalReward=100, stepLoss=-1)

#env = gym.make('CartPole-v1')
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
#    from IPython import display
# plt.ion()

"""class neural_network(torch.nn.Module):
    def __init__(self, observation_space_n, action_space_n):
        super(neural_network, self).__init__()
        self.layer1 = torch.nn.Linear(observation_space_n, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, action_space_n)

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)"""

class neural_network(torch.nn.Module):
    def __init__(self, observation_space_n, action_space_n):
        super(neural_network, self).__init__()
 
        channels = 3
        size = int(math.sqrt(observation_space_n / channels)) # size 8 makes observation_space_n 8x8x3=192
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.flattened_size = 32 * size * channels
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, action_space_n)

    def forward(self, x):
        # Ensure the input has the correct shape [batch_size, channels, height, width]
        #x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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

class DQNAgentConv():
    def __init__(self, action_space, observation, batch_size=128, lr=0.0001, gamma=0.99, epsilon_start=0.9, epsilon_min=0.05, epsilon_decay=1000, tau=0.005, replayBuffer=10_000, wandb=None):
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
        self.memory = ReplayMemory(replayBuffer)
        self.steps_done = 0
        self.episode_rewards = []
        self.rewardPerStep_history = []
        self.epsilon_history = []
        self.loss = 0.0
        self.loss_history = []
        self.gradients_mean_history = []
        self.gradients_stdev_history = []


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
        rewardPerStep_t = torch.tensor(self.rewardPerStep_history, dtype=torch.float)
        epsilons_t = torch.tensor(self.epsilon_history, dtype=torch.float)
        loss_t = torch.tensor(self.loss_history, dtype=torch.float)
        gradients_mean_t = torch.tensor(self.gradients_mean_history, dtype=torch.float)
        gradients_stdev_t = torch.tensor(self.gradients_stdev_history, dtype=torch.float)
        
        if show_result:
            plt.title('Result')
        else:
            plt.clf() # Clear the figure
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.plot(rewards_t.numpy(), label='Ep Reward')
        plt.plot(epsilons_t.numpy(), label='Epsilon')
        plt.plot(loss_t.numpy(), label='Loss')
        plt.plot(gradients_mean_t.numpy(), label='Gradients mean')
        plt.plot(gradients_stdev_t.numpy(), label='Gradients stdev')

        # Plot 100 episode averages
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label='100-ep av reward')
            rps_means = rewardPerStep_t.unfold(0, 100, 1).mean(1).view(-1)
            rps_means = torch.cat((torch.zeros(99), rps_means))
            plt.plot(rps_means.numpy(), label='100-ep av reward/steps')
        #plt.plot(rewardPerStep_t.numpy(), label='Reward/steps')

        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated
        if not show_result:
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

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
        
        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()

        # In-place gradient clipping (todo: maybe remove this)
        #torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

    def train(self, env, num_episodes=200):
        # Loop through episodes
        timeStart = time.time()
        for i_episode in range(num_episodes):
            if i_episode == num_episodes // 100:
                elapsed_time = time.time() - timeStart
                finish_time = time.time() + elapsed_time * 99
                
                # Convert elapsed time to minutes and seconds
                minutes, seconds = divmod(elapsed_time, 60)
                
                # Convert finish time to a readable time of day
                finish_time_readable = datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"1%, time elapsed: {int(minutes)} minutes and {seconds:.2f} seconds, it may finish around: {finish_time_readable}")
            
            # 10% time estimate
            if i_episode == num_episodes // 10:
                elapsed_time = time.time() - timeStart
                finish_time = time.time() + elapsed_time * 9
                
                # Convert elapsed time to minutes and seconds
                minutes, seconds = divmod(elapsed_time, 60)
                
                # Convert finish time to a readable time of day
                finish_time_readable = datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"10%, time elapsed: {int(minutes)} minutes and {seconds:.2f} seconds, it may finish around: {finish_time_readable}")
            # 50% time estimate
            if i_episode == num_episodes // 2:
                elapsed_time = time.time() - timeStart
                finish_time = time.time() + elapsed_time
                
                # Convert elapsed time to minutes and seconds
                minutes, seconds = divmod(elapsed_time, 60)
                
                # Convert finish time to a readable time of day
                finish_time_readable = datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"50%, time elapsed: {int(minutes)} minutes and {seconds:.2f} seconds, it may finish around: {finish_time_readable}")
            
            # Init env and git its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) #.unsqueeze(1)
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
                        device=self.device).unsqueeze(0) #.unsqueeze(1)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization
                self.optimize_model()

                """
                # Maybe change to updated around every 2000 steps instead of soft update
                if i_episode % 30 == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                """
                
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
                    self.rewardPerStep_history.append(episode_reward / (i + 1))
                    self.epsilon_history.append(epsilon)
                    self.loss_history.append(self.loss)

                    gradients = []
                    for p in self.policy_network.parameters():
                        if p.grad is not None:
                            gradients.extend(p.grad.data.cpu().numpy().flatten())  # Collect all gradient values
                    gradients_mean = np.mean(gradients)
                    gradients_std = np.std(gradients)
                    self.gradients_mean_history.append(gradients_mean)
                    self.gradients_stdev_history.append(gradients_std)
                    
                    if self.wandb:
                        self.wandb.log({"episode_reward": episode_reward,
                                    "epsilon": epsilon,
                                    "rewardPerStep": episode_reward / (i + 1),
                                    "loss": self.loss,
                                    "gradient mean": gradients_mean,
                                    "gradient std": gradients_std})
                    self.plot_rewards()
                    break
            
        print("Complete")
        self.plot_rewards()
        plt.ioff()
        plt.show()

    def save_model_weights(self, path):
        torch.save(self.policy_network.state_dict(), path)
        print(f"Model saved: {path}")

    def load_model_weights(self, path):
        self.policy_network.load_state_dict(torch.load(path))
        self.policy_network.eval()
        print(f"Model loaded: {path}")

    def inference(self, env, num_episodes=200, epsilon=0.05):
        for _ in range(num_episodes):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) #.unsqueeze(1)
            episode_reward = 0
            for i in count():
                if epsilon > random.random():
                    action = torch.tensor([[random.randrange(self.action_space.n)]], device=self.device, dtype=torch.long)
                else:
                    with torch.no_grad():
                        action = self.policy_network(state).max(1)[1].view(1, 1)
                
                observation, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32,
                        device=self.device).unsqueeze(0) #.unsqueeze(1)

                state = next_state

                if done:
                    self.episode_rewards.append(episode_reward)
                    self.loss_history.append(self.loss)
                    #self.plot_rewards()
                    break
            
        print("Complete")
        #self.plot_rewards(show_result=True)
        #plt.ioff()
        #plt.show()
    
    def get_q_values(self, env, observation):
        
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        print(f"action nr: {self.policy_network(observation).max(1)[1].view(1, 1)}")
        print(f"Q values: {self.policy_network(observation)}")
        print(f"Max Q value: {self.policy_network(observation).max().item()}")

        return self.policy_network(observation).max(1)[1].view(1, 1)

# main
if __name__ == "__main__":

    preName = "3Big02Coins0walls" #"PreTrainedConv2RandAbs3walls0to1"   #+ "_6x6_3000episodes"

    # Config
    num_episodes = 12_000

    # DQNGridWorldEnv
    size=8
    agentSpawn=None
    targetSpawn=None
    goalReward=1
    stepLoss=-0.005
    maxSteps=200
    wallCoordinates=None
    forbiddenCoordinates=None
    forbiddenPenalty=-0.4
    chanceOfSupervisor=[0.0, 1]
    randomWalls=0
    randomForbiddens=0
    goodCoinCoordinates=None
    badCoinCoordinates=None
    goodCoinReward=0.2
    badCoinPenalty=-0.05
    randomGoodCoins=3
    randomBadCoins=0

    # Agent
    useWandb = True
    batch_size=64
    lr=0.001
    gamma=0.95
    epsilon_start=1
    epsilon_min=0.05
    epsilon_decay=200_000 # 50_000 at 3000 episodes
    tau=0.0005 # Was 0.005
    replayBuffer=100_000

    if useWandb:
        wandb.init(project=f"{preName}_{size}x{size}_{num_episodes}episodes",
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
            "tau": tau})
    else:
        wandb = None

    env = DQNGridWorldEnvConv(render_mode=None, size=size, agentSpawn=agentSpawn, targetSpawn=targetSpawn, goalReward=goalReward, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates, forbiddenCoordinates=forbiddenCoordinates, forbiddenPenalty=forbiddenPenalty, chanceOfSupervisor=chanceOfSupervisor, randomWalls=randomWalls, randomForbiddens=randomForbiddens, goodCoinCoordinates=goodCoinCoordinates, badCoinCoordinates=badCoinCoordinates, goodCoinReward=goodCoinReward, badCoinPenalty=badCoinPenalty, randomGoodCoins=randomGoodCoins, randomBadCoins=randomBadCoins)
    observation, info = env.reset()
    agent = DQNAgentConv(env.action_space, observation,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        tau=tau,
        replayBuffer=replayBuffer,
        wandb=wandb)

    # Todo:
    # - normalize, check!
    # - plot everything (plot gradients, should be normal dist around 0)
    # - remove clipping, check!


    
    #agent.load_model_weights(f"C:/Projects/public/XAI_NTNU/models/{size}x{size}_{num_episodes}ep.pth")
    print(f"First observation:\n {observation}")
    print(f"First observation.shape: {observation.shape}")
    # agent.load_model_weights(f"C:/Projects/public/XAI_NTNU/models/2goodCoins0walls8x8_3000ep.pth")
    agent.train(env=env, num_episodes=num_episodes)
    # chanceOfSupervisor=[0.0, 1]
    if preName is not None:
        agent.save_model_weights(f"C:/Projects/public/XAI_NTNU/models/{preName}_{size}x{size}_{num_episodes}ep.pth")
    show_env = DQNGridWorldEnvConv(render_mode="human", size=size, agentSpawn=None, targetSpawn=targetSpawn, goalReward=goalReward, stepLoss=stepLoss, maxSteps=20, wallCoordinates=wallCoordinates, forbiddenCoordinates=forbiddenCoordinates, forbiddenPenalty=forbiddenPenalty, chanceOfSupervisor=chanceOfSupervisor, randomWalls=randomWalls, randomForbiddens=randomForbiddens, goodCoinCoordinates=goodCoinCoordinates, badCoinCoordinates=badCoinCoordinates, goodCoinReward=goodCoinReward, badCoinPenalty=badCoinPenalty, randomGoodCoins=randomGoodCoins, randomBadCoins=randomBadCoins)
    agent.inference(env=show_env, num_episodes=50, epsilon=epsilon_min)