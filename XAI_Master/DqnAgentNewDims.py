import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from collections import deque
import time
from datetime import datetime
import random
import wandb
import shap
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings

from WallWorld import WallWorld

class neural_network(torch.nn.Module):
    def __init__(self, observation, action_space_n):
        super(neural_network, self).__init__()
        channels = observation.shape[0] # 3 channels (RGB)
        height = observation.shape[1]
        width = observation.shape[2]

        # Expoecting shape is [batch_size, 3, 7, 7]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.flattened_size = 32 * height * width

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, action_space_n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DqnAgentNewDims:
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
        maxReward = -2 # init number under minimum reward
        minReward = 2 # init number over maximum reward
        while self.steps_done < max_steps:
            episodeSteps, episodeReward = 0, 0
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            terminated = False
            truncated = False
            try:
                while not terminated and not truncated:
                    if self.steps_done in [int(x*max_steps) for x in [0.01, 0.1, 0.2, 0.5, 0.75, 0.99]]: # print percentages
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
                print("Interrupted")
                break
    
    def inference(self, env, max_steps=1_000_000, epsilon=0.05, renderQvalues=False):
        self.train(env, max_steps, train=False, fixedEpsilon=epsilon, renderQvalues=renderQvalues)
    
    def createActivationDataset(self, env, num_episodes=100, epsilon=0.05):
        df = pd.DataFrame(columns=["target"]
                        + [f'{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']]
                        + [f'a{i}' for i in range(1, 129)])
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        # Register hook on the second last layer (fc1)
        hook = self.policy_net.fc1.register_forward_hook(get_activation('fc1'))

        for e in range(num_episodes):
            state, _ = env.reset()
            # agentX, agentY = env.agentCoordinates, old
            # redX, redY = env.redChestCoordinates[0], old
            # greenX, greenY = env.greenChestCoordinates[0], old
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state_init = state.clone()
            terminated = False
            truncated = False

            # Take 1 step to get the activations
            action = self.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state = next_state
            activationData = activations['fc1'].cpu().numpy()[0].flatten()

            while not terminated and not truncated:
                if np.random.rand() < epsilon:
                    action = torch.tensor([[random.randrange(self.action_space.n)]], device=self.device, dtype=torch.long)
                else:
                    with torch.no_grad():
                        action = self.policy_net(state).max(1)[1].view(1, 1)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                state = next_state
            if terminated:
                target = reward.item()          # 1 reward ---> green chest
            else:                               # -1 reward ---> red chest
                target = 0                      # 0 truncated --> no chest
            df.loc[e] = [target] + list(state_init.flatten().detach().numpy()) + list(activationData)
        hook.remove()
        return df
    
    def createUncoloredState(self, state):
        uncolored_state = state.clone()
        height, width = state.shape[1:3]
        for i in range(height):
            for j in range(width):
                # Check if the cell has any non-zero values
                if torch.any(state[0][i][j] > 0):
                    # Set to white using tensor values
                    uncolored_state[0][i][j][:] = 1.0
        return uncolored_state

    def createGradCamDataset(self, env, num_episodes=100, epsilon=0.05):
        df = pd.DataFrame(columns=["target"]
                        + [f'{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']]
                        + [f'gc{i}{j}' for i in range(7) for j in range(7)])
        
        for e in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state_init = state.clone()
            action = self.policy_net(state).max(1)[1].view(1, 1)
 
            with GradCAM(model=self.policy_net, target_layers=[self.policy_net.conv3]) as cam:
                grayscaleGradCAM = cam(input_tensor=state, targets=[ClassifierOutputTarget(action.item())])[0, :]
                

            terminated = False
            truncated = False

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state = next_state

            while not terminated and not truncated:
                if np.random.rand() < epsilon:
                    action = torch.tensor([[random.randrange(self.action_space.n)]], device=self.device, dtype=torch.long)
                else:
                    with torch.no_grad():
                        action = self.policy_net(state).max(1)[1].view(1, 1)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                state = next_state
            if terminated:
                target = reward.item()          # 1 reward ---> green chest
            else:                               # -1 reward ---> red chest
                target = 0                      # 0 truncated --> no chest

            df.loc[e] = [target] + list(state_init.flatten().detach().numpy()) + list(grayscale_cam.flatten())
        return df

    def createShapDataset(self, env, backgroundData=None, batch_size=64, num_episodes=100, epsilon=0.05):
        if backgroundData is None:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            backgroundData = self.createUncoloredState(state)
        df = pd.DataFrame(columns=["target"]
                        + [f'{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']]
                        + [f'shap{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']])
        
        for e in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state_init = state.clone()
            action = self.policy_net(state).max(1)[1].view(1, 1)
            shap_values = shap.GradientExplainer(self.policy_net, backgroundData, batch_size=batch_size).shap_values(state)
            action_shap_values = np.array([s[:,:,:,action.item()] for s in shap_values])

            terminated = False
            truncated = False

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state = next_state

            while not terminated and not truncated:
                if np.random.rand() < epsilon:
                    action = torch.tensor([[random.randrange(self.action_space.n)]], device=self.device, dtype=torch.long)
                else:
                    with torch.no_grad():
                        action = self.policy_net(state).max(1)[1].view(1, 1)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                state = next_state
            if terminated:
                target = reward.item()          # 1 reward ---> green chest
            else:                               # -1 reward ---> red chest
                target = 0                      # 0 truncated --> no chest

            df.loc[e] = [target] + list(state_init.flatten().detach().numpy()) + list(action_shap_values.flatten())
        return df
    
    def createFullDataset(self, env, backgroundData=None, batch_size=64, num_episodes=100, epsilon=0.05, steps=4):
        if backgroundData is None:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            backgroundData = self.createUncoloredState(state)
        df = pd.DataFrame(columns=["target"] + ["stepsTaken"]
                        + [f'pix{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']]
                        + [f'act{i}' for i in range(1, 129)]
                        + [f'shap{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']]
                        + [f'ngc{i}{j}' for i in range(7) for j in range(7)]
                        + [f'xgc{i}{j}' for i in range(7) for j in range(7)]
                        + [f'gcpp{i}{j}' for i in range(7) for j in range(7)])
        
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        # Register hook on the second last layer (fc1)
        hook = self.policy_net.fc1.register_forward_hook(get_activation('fc1'))

        episode_df = pd.DataFrame(columns=["target"] + ["stepsTaken"]
                        + [f'pix{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']]
                        + [f'act{i}' for i in range(1, 129)]
                        + [f'shap{i}{j}{c}' for i in range(7) for j in range(7) for c in ['r','g','b']]
                        + [f'ngc{i}{j}' for i in range(7) for j in range(7)]
                        + [f'xgc{i}{j}' for i in range(7) for j in range(7)]
                        + [f'gcpp{i}{j}' for i in range(7) for j in range(7)])
        for e in range(num_episodes):
            # remove all rows from episode_df
            episode_df = episode_df.iloc[0:0]
            step_count = 0

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.policy_net(state).max(1)[1].view(1, 1)

            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                if step_count <= steps:
                    activationData = activations['fc1'].cpu().numpy()[0].flatten()
                    shap_values = shap.GradientExplainer(self.policy_net, backgroundData, batch_size=batch_size).shap_values(state)
                    action_shap_values = np.array([s[:,:,:,action.item()] for s in shap_values])
                    with GradCAM(model=self.policy_net, target_layers=[self.policy_net.conv3]) as cam:
                        grayscaleGradCAM = cam(input_tensor=state, targets=[ClassifierOutputTarget(action.item())])[0, :]
                    with XGradCAM(model=self.policy_net, target_layers=[self.policy_net.conv3]) as cam:
                        grayscaleXGradCAM = cam(input_tensor=state, targets=[ClassifierOutputTarget(action.item())])[0, :]
                    with GradCAMPlusPlus(model=self.policy_net, target_layers=[self.policy_net.conv3]) as cam:
                        grayscaleGradCAMplusPlus = cam(input_tensor=state, targets=[ClassifierOutputTarget(action.item())])[0, :]
                    #GradCAMPlusPlus

                    episode_df.loc[step_count] = [0] + [step_count] + list(state.flatten().detach().numpy()) + list(activationData) + list(action_shap_values.flatten()) + list(grayscaleGradCAM.flatten()) + list(grayscaleXGradCAM.flatten() + list(grayscaleGradCAMplusPlus.flatten()))
                    step_count += 1
            
                if np.random.rand() < epsilon:
                    action = torch.tensor([[random.randrange(self.action_space.n)]], device=self.device, dtype=torch.long)
                else:
                    with torch.no_grad():
                        action = self.policy_net(state).max(1)[1].view(1, 1)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                state = next_state

            if terminated:
                target = reward.item()          # 1 reward ---> green chest
            else:                               # -1 reward ---> red chest
                target = 0                      # 0 truncated --> no chest
            # set target in all rows in episode_df to target
            episode_df["target"] = target
            # append episode_df to df
            if not episode_df.empty and not episode_df.isna().all(axis=None):
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame concatenation with empty or all-NA entries.*")
                df = pd.concat([df, episode_df], ignore_index=True)
            else:
                print(f"Episode {e}, step {step_count} had a NaN :(")
        hook.remove()
        return df

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
        if not percent > 0:
            return
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
    max_steps=3_000_000

    # WallWorld
    render_mode=None
    size=7
    agentSpawn = None
    maxSteps=200
    stepLoss=0
    chestSpawnCoordinates=np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
                                    [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                                    [2, 0], [2, 1], [2, 2],         [2, 4], [2, 5], [2, 6]])
    wallCoordinates=      np.array([[3, 0], [3, 1], [3, 2],         [3, 4], [3, 5], [3, 6],])
    agentSpawnCoordinates=np.array([[4, 0],                                         [4, 6],
                                    [5, 0],                                         [5, 6],
                                    [6, 0],                                         [6, 6]])
    randomWalls=0
    redChestCoordinates=None
    greenChestCoordinates=None
    keyCoordinates=None
    randomredChests=1
    randomgreenChests=1
    randomkeys=0
    
    # Agent
    batch_size=64
    lr=0.001
    gamma=0.95
    epsilon_start=1
    epsilon_min=0.05
    epsilon_decay=200_000 # 50_000 at 3000 episodes
    tau=0.0005 # Was 0.005
    replayBuffer=100_000

    rewardCombinations = [[0.0, 1],
               [0.1, 1],
               [0.2, 1],
               [0.3, 1],
               [0.4, 1],
               [0.5, 1],
               [0.6, 1],
               [0.7, 1],
               [0.8, 1],
               [0.9, 1],
               [1, 1],
               [1, 0.9],
               [1, 0.8],
               [1, 0.7],
               [1, 0.6],
               [1, 0.5],
               [1, 0.4],
               [1, 0.3],
               [1, 0.2],
               [1, 0.1],
               [1, 0.0]]
    """
    
    for rewardCombination in rewardCombinations:
        print(f"Reward combination: {rewardCombination}")
        print(f"Reward combination[0]: {rewardCombination[0]}")
        print(f"Reward combination[1]: {rewardCombination[1]}")
        
        redChestReward=rewardCombination[0] # Changes during evaluation
        greenChestReward=rewardCombination[1] # Changes during evaluation

        model_name = f"r{str(int(redChestReward * 10)).zfill(2)}_g{str(int(greenChestReward * 10)).zfill(2)}_{max_steps//1000}k"

        useWandb = True
        saveModel = True

        if useWandb:
            wandb.init(project=model_name,
                config={
                "size": size,
                "redChestReward": redChestReward,
                "greenChestReward": greenChestReward,
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

        env = WallWorld(render_mode=None,
            size=size, agentSpawn=agentSpawn,
            stepLoss=stepLoss, maxSteps=maxSteps,
            wallCoordinates=wallCoordinates,
            randomWalls=randomWalls,
            redChestCoordinates=redChestCoordinates,
            greenChestCoordinates=greenChestCoordinates,
            keyCoordinates=keyCoordinates,
            redChestReward=redChestReward,
            greenChestReward=greenChestReward,
            randomredChests=randomredChests,
            randomgreenChests=randomgreenChests,
            randomkeys=randomkeys,
            agentSpawnCoordinates=agentSpawnCoordinates,
            chestSpawnCoordinates=chestSpawnCoordinates,
            newDims=True)
        observation, _ = env.reset()
        agent = DqnAgentNewDims(env.action_space, observation,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            tau=tau,
            replayBuffer=replayBuffer,
            wandb=wandb)
        #agent.load_model_weights(f"C:/Projects/public/XAI_Master/models/WW_redReward1_grReward0_7x7_2000000steps.pth")
        
        agent.train(env=env, max_steps=max_steps)
        if saveModel:
            agent.save_model_weights(f"C:/Projects/public/XAI_Master/models/y_{model_name}.pth")
    print("Complete")

    """
    env = WallWorld(render_mode=None,
            size=size, agentSpawn=agentSpawn,
            stepLoss=stepLoss, maxSteps=maxSteps,
            wallCoordinates=wallCoordinates,
            randomWalls=randomWalls,
            redChestCoordinates=redChestCoordinates,
            greenChestCoordinates=greenChestCoordinates,
            keyCoordinates=keyCoordinates,
            redChestReward=1,
            greenChestReward=1,
            randomredChests=randomredChests,
            randomgreenChests=randomgreenChests,
            randomkeys=randomkeys,
            agentSpawnCoordinates=agentSpawnCoordinates,
            chestSpawnCoordinates=chestSpawnCoordinates,
            newDims=True)
    observation, _ = env.reset()

    maxSteps = 200
    
    observation, _ = env.reset()
    print(f"env.action_space: {env.action_space}")
    print(f"observation: {observation.shape}")
    print(f"observation: {observation}")
    agent = DqnAgentNewDims(env.action_space, observation,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        tau=tau,
        replayBuffer=replayBuffer)

    useWandb = False

    model_name = "newDimsModel_2000k"

    if useWandb:
        wandb.init(project=model_name,
            config={
            "size": size,
            "redChestReward": 1,
            "greenChestReward": 1,
            "stepLoss": stepLoss,
            "maxSteps": maxSteps,
            "batch_size": batch_size,
            "lr": lr,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "tau": tau})


    #agent.train(env=env, max_steps=2_000_000)
    
    agent.load_model_weights(f"C:/Projects/public/XAI_Master/models/r10_g10_3000k.pth")

    maxSteps = 30
    show_env = WallWorld(render_mode="human",
        size=size, agentSpawn=agentSpawn,
        stepLoss=stepLoss, maxSteps=maxSteps,
        wallCoordinates=wallCoordinates,
        randomWalls=randomWalls,
        redChestCoordinates=redChestCoordinates,
        greenChestCoordinates=greenChestCoordinates,
        keyCoordinates=keyCoordinates,
        redChestReward=1,
        greenChestReward=1,
        randomredChests=randomredChests,
        randomgreenChests=randomgreenChests,
        randomkeys=randomkeys,
        agentSpawnCoordinates=agentSpawnCoordinates,
        chestSpawnCoordinates=chestSpawnCoordinates,
        newDims=True)
    
    agent.inference(env=show_env, max_steps=1_000_000, epsilon=epsilon_min, renderQvalues=False)



