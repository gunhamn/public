
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import deque
import time
from datetime import datetime
import random
import wandb

from ChestWorld import ChestWorld

# A DQN is a RL-agent that takes action in an enviroment where it at some point gains rewards from the environment.
# D stands for Deep as in Deep Learning, which means that the algorithm uses a neural network to approximate q-values
# Q-values is the approximated value of a state, and every timestep the agent either performs the action
# with the highest approximated q-values, or a random action with the probabilty epsilon.
# Epsilon often start close to 1 and end up around 0,05. Meaning that the trained agent still performs
# random actions every 20th timestep, since it should keep its epsilon also in the deployment environment.

# A DQN uses 2 networks. They are called a target and a policy network. The policy network is updated every timestep
# while the target network is only copied from the policy network every perhaps 1000nd timestep. Having these 2 networks
# instead of one increases the algorithms stability.
# When updating the policy network, the algorithm takes in a big batch of so-called transitions. A transition is an
# tuple with (state, action, next_state, reward) where, in Gym, state and next_state are (observation, reward, terminated, True, info).
# Every gamestep the agent has seen is stored in its buffer of transitions.
# It has a big buffer size that removes the oldest transitions continously when its reaches its max size.
# A rndomly sampled batch of transitions are used to train the policy network every timestep.

# The q-value is calculated by the bellman equation

# Q(s, a) = r + discountFactor * max[Q(s', a')]

# Where:
# Q(s, a) - Q-value for s, a - state, action
# r - reward
# discountFactor, a float like 0.95
# max[Q(s', a')] the maximum q-value for all actions a' in the next state 's

# Meaning, no reward now but a win with reward 1 after 7 actions yields
# Q(s, a) = 0 + 0.95^7 * 1 = 0.698




"""
Components

Replay buffer
Neural net
Epsilon greedy
Agent
    Init
    Train
    Optimise
    Inference

"""
class neural_network(torch.nn.Module):
    def __init__(self, observation_space_n, action_space_n):
        super(neural_network, self).__init__()
        print(f"Action space: {action_space_n}")
        # Action space: 4
        channels = 3
        size = len(observation_space_n)
        print(f"Size: {size}")
        # Size: 6
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.flattened_size = 32 * size * channels
        print(f"Flattened size: {self.flattened_size}")
        # Flattened size: 576
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, action_space_n)

    def forward(self, x):
        # Ensure the input has the correct shape [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DqnAgent:
    def __init__(self, action_space, observation, batch_size=128, lr=0.0001, gamma=0.99, epsilon_start=0.9, epsilon_min=0.05, epsilon_decay=1000, tau=0.001, replayBuffer=10_000, trainFrequency=15, wandb=None):
        self.action_space = action_space
        self.observation = observation
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.updateTargetNetInterval = int(1 / self.tau) # just to keep the same input format as the last dqn agent
        self.replayBuffer = replayBuffer
        self.wandb = wandb
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")
        self.replay_buffer = deque(maxlen=replayBuffer)
        self.trainFrequency = trainFrequency

        self.policy_net = neural_network(observation, action_space.n).to(self.device)
        self.target_net = neural_network(observation, action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        random.seed(48)
        np.random.seed(48)
        torch.manual_seed(48)

    def train(self, env, max_steps=1_000_000, train=True, fixedEpsilon=None, renderQvalues=False):
        timeStart = time.time()
        self.steps_done = 0
        wandFrequency = 1000
        loss = torch.tensor([0], dtype=torch.float32, device=self.device) # for plotting in wandb
        lastEpisodeReward, cummulativeReward, cummulativeSteps = 0, 0, 0
        maxReward = -2 # a number under minimum reward
        minReward = 2 # a number over maximum reward
        while self.steps_done < max_steps:
            episodeSteps, episodeReward = 0, 0
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            terminated = False
            truncated = False
            try:
                while not terminated and not truncated:
                    if self.steps_done in [int(x*max_steps) for x in [0.01, 0.1, 0.5, 0.99]]: # print percentages
                        self.printProgress(timeStart, round(self.steps_done/max_steps, 2))

                    epsilon = fixedEpsilon or self.epsilon_exp_decay(max_steps)
                    if np.random.rand() < epsilon:
                        action = torch.tensor([[random.randrange(self.action_space.n)]], device=self.device, dtype=torch.long)
                    else:
                        with torch.no_grad():
                            action = self.policy_net(state).max(1)[1].view(1, 1)
                    
                    if renderQvalues:
                        env.q_values = self.getQvalues(env, normalize=True)
                        env.q_value_actions = self.getQvalueActions(env)
                    next_state, reward, terminated, truncated, _ = env.step(action.item())

                    # For plotting in wandb
                    cummulativeReward += reward
                    episodeReward += reward
                    if terminated or truncated:
                        lastEpisodeReward = episodeReward
                        maxReward = max(maxReward, episodeReward)
                        minReward = min(minReward, episodeReward)
                    
                    reward = torch.tensor([reward], device=self.device)
                    terminated = torch.tensor([terminated], dtype=torch.bool, device=self.device)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                    # Handle truncation  here?

                    self.replay_buffer.append((state, action, next_state, reward, terminated))
                    
                    state = next_state
                    self.steps_done += 1
                    episodeSteps += 1
                    cummulativeSteps += 1

                    if self.steps_done % self.trainFrequency == 0 and train:
                        if len(self.replay_buffer) >= self.batch_size:
                            # Sample a batch of transitions
                            batch = random.sample(self.replay_buffer, self.batch_size)

                            # Unpack the batch
                            states, actions, next_states, rewards, terminateds = zip(*batch)

                            # Convert the batch to tensors
                            states = torch.cat(states) # float32, torch.Size([64, 8, 8, 3])
                            actions = torch.cat(actions) # int64, torch.Size([64, 1])
                            next_states = torch.cat(next_states) # float32, torch.Size([64, 8, 8, 3])
                            rewards = torch.cat(rewards) # float32, torch.Size([64])
                            terminateds = torch.cat(terminateds).to(torch.int8).to(self.device) # was bool, now int8, torch.Size([64])

                            # Q(s, a) = r + discountFactor * max[Q(s', a')], only return reward if terminated
                            with torch.no_grad():
                                targets = rewards + self.gamma * self.target_net(next_states).max(1)[0] * (1 - terminateds) # float32, torch.Size([64])
                            predictions = self.policy_net(states).gather(1, actions).squeeze() # float32, torch.Size([64])
                            loss = nn.functional.mse_loss(predictions, targets)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            
                    if self.steps_done % self.updateTargetNetInterval == 0 and train:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                    if self.steps_done % wandFrequency == 0 and self.wandb:
                        self.wandb.log({
                            "epsilon": epsilon,
                            "loss": loss.item(),
                            "reward/episodeSteps": cummulativeReward / cummulativeSteps,
                            "lastEpisodeReward": lastEpisodeReward,
                            f"maxReward, {wandFrequency} steps": maxReward,
                            f"minReward, {wandFrequency} steps": minReward
                        })
                        cummulativeReward, cummulativeSteps, maxReward, minReward = 0, 0, -2, 2

            # Ctrl+C to evaluate agent
            except KeyboardInterrupt:
                print("Training interrupted")
                break
    
    def inference(self, env, max_steps=1_000_000, epsilon=0.05, renderQvalues=False):
        self.train(env, max_steps, train=False, fixedEpsilon=epsilon, renderQvalues=renderQvalues)
        
    def predict(self, state):
        with torch.no_grad():
            return self.policy_net(state)

    def epsilon_exp_decay(self, max_steps):
        decay_rate = np.log(self.epsilon_min / self.epsilon_start) / max_steps
        return self.epsilon_start * np.exp(decay_rate * self.steps_done)
    
    def getQvalues(self, env, normalize=True):
        qValues = np.zeros((env.size, env.size))
        for i in range(len(qValues)):
            for j in range(len(qValues[i])):
                state = env._get_obs(agentLoc=np.array([i, j]))
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                qValues[i][j] = self.predict(state).max(1)[0].item()
        if normalize:
            qValues = (qValues - qValues.min()) / (qValues.max() - qValues.min())
        return qValues

    def getQvalueActions(self, env):
        qValueActions = np.zeros((env.size, env.size))
        for i in range(len(qValueActions)):
            for j in range(len(qValueActions[i])):
                state = env._get_obs(agentLoc=np.array([i, j]))
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                qValueActions[i][j] = self.predict(state).max(1)[1].item()
        return qValueActions
            
    def printProgress(self, timeStart, percent):
        elapsed_time = time.time() - timeStart
        est_finish_time = time.time() + (elapsed_time / percent) * (1 - percent)
        minutes, seconds = divmod(elapsed_time, 60)
        finish_time_readable = datetime.fromtimestamp(est_finish_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{int(percent*100)}%, time elapsed: {int(minutes)} minutes and {seconds:.2f} seconds, it may finish around: {finish_time_readable}")

    def save_model_weights(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved: {path}")

    def load_model_weights(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        print(f"Model loaded: {path}")




if __name__ == "__main__":

    # Config
    max_steps=1000_000

    # ChestWorld
    render_mode=None
    size=7
    agentSpawn = None
    maxSteps=200
    stepLoss=-1/maxSteps # min reward should be -1
    wallCoordinates=np.array([[3, 0],[3, 1],[3, 2],     [3, 4],[3, 5],[3, 6],])
    randomWalls=0
    chestCoordinates=None
    keyCoordinates=None
    randomchests=1
    randomkeys=0
    chestReward=1
    
    # Agent
    batch_size=64
    lr=0.001
    gamma=0.95
    epsilon_start=1
    epsilon_min=0.05
    epsilon_decay=200_000 # 50_000 at 3000 episodes
    tau=0.0005 # Was 0.005
    replayBuffer=100_000

    model_name = f"CW_{randomchests}chests_{randomkeys}keys_{randomWalls}walls_{size}x{size}_{max_steps}steps"

    useWandb = False
    saveModel = False

    if useWandb:
        wandb.init(project=model_name,
            config={
            "size": size,
            "goalReward": chestReward,
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
        wandb = False

    env = ChestWorld(render_mode=None, size=size, agentSpawn=agentSpawn, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates, randomWalls=randomWalls, chestCoordinates=chestCoordinates, keyCoordinates=keyCoordinates, chestReward=chestReward, randomchests=randomchests, randomkeys=randomkeys)
    observation, _ = env.reset()
    agent = DqnAgent(env.action_space, observation,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        tau=tau,
        replayBuffer=replayBuffer,
        wandb=wandb)
    
    #agent.load_model_weights(f"C:/Projects/public/XAI_NTNU/models/{size}x{size}_{num_episodes}ep.pth")
    print(f"First observation:\n {observation}")
    print(f"First observation.shape: {observation.shape}")
    #agent.load_model_weights(f"C:/Projects/public/XAI_NTNU/modelsToEval/CW_3chests_5keys_6x6_10000000steps.pth")
    print(f"q values: {agent.getQvalues(env)}")
    print(f"q value actions: {agent.getQvalueActions(env)}")
    # agent.load_model_weights(f"C:/Projects/public/XAI_NTNU/models/2goodCoins0walls8x8_3000ep.pth")
    agent.train(env=env, max_steps=max_steps)
    if saveModel:
        agent.save_model_weights(f"C:/Projects/public/XAI_NTNU/models/{model_name}.pth")
    maxSteps = 30
    show_env = ChestWorld(render_mode="human", size=size, agentSpawn=agentSpawn, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates, randomWalls=randomWalls, chestCoordinates=chestCoordinates, keyCoordinates=keyCoordinates, chestReward=chestReward, randomchests=randomchests, randomkeys=randomkeys)
    agent.wandb = False
    agent.inference(env=show_env, max_steps=1000, epsilon=epsilon_min, renderQvalues=False)
