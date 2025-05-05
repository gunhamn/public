import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap
import time

from WallWorld import WallWorld
from DqnAgentNewDims import DqnAgentNewDims

if __name__ == "__main__":

    # Config
    max_steps=500_000

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
    redChestReward=-1 # Don't change this
    greenChestReward=1 # Don't change this
    # Explaination:
    # terminated with 1 reward ---> green chest: 1
    # terminated with 0 reward ---> red chest:  -1
    # truncated                 --> no chest:    0

    # Agent
    batch_size=64
    lr=0.001
    gamma=0.95
    epsilon_start=1
    epsilon_min=0.05
    epsilon_decay=200_000 # 50_000 at 3000 episodes
    tau=0.0005 # Was 0.005
    replayBuffer=100_000

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
        replayBuffer=replayBuffer)
    
    maxSteps = 30 #???? Should be 200??
    show_env = WallWorld(render_mode="human",
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
    """
    modelNames = ["r00_g10_1500k",
              "r01_g10_1500k",
              "r02_g10_1500k",
              "r03_g10_1500k",
              "r04_g10_1500k",
              "r05_g10_1500k",
              "r06_g10_1500k",
              "r07_g10_1500k",
              "r08_g10_1500k",
              "r09_g10_1500k",
              "r10_g00_1500k",
              "r10_g01_1500k",
              "r10_g02_1500k",
              "r10_g03_1500k",
              "r10_g04_1500k",
              "r10_g05_1500k",
              "r10_g06_1500k",
              "r10_g07_1500k",
              "r10_g08_1500k",
              "r10_g09_1500k",
              "r10_g10_1500k"]
    """
    
    
    modelNames = ["r00_g10_3000k",
              "r01_g10_3000k",
              "r02_g10_3000k",
              "r03_g10_3000k",
              "r04_g10_3000k",
              "r05_g10_3000k",
              "r06_g10_3000k",
              "r07_g10_3000k",
              "r08_g10_3000k",
              "r09_g10_3000k",
              "r10_g00_3000k",
              "r10_g01_3000k",
              "r10_g02_3000k",
              "r10_g03_3000k",
              "r10_g04_3000k",
              "r10_g05_3000k",
              "r10_g06_3000k",
              "r10_g07_3000k",
              "r10_g08_3000k",
              "r10_g09_3000k",
              "r10_g10_3000k"]

    y_modelNames = ["y_r00_g10_3000k",
                "y_r01_g10_3000k",
                "y_r02_g10_3000k",
                "y_r03_g10_3000k",
                "y_r04_g10_3000k",
                "y_r05_g10_3000k",
                "y_r06_g10_3000k",
                "y_r07_g10_3000k",
                "y_r08_g10_3000k",
                "y_r09_g10_3000k",
                "y_r10_g00_3000k",
                "y_r10_g01_3000k",
                "y_r10_g02_3000k",
                "y_r10_g03_3000k",
                "y_r10_g04_3000k",
                "y_r10_g05_3000k",
                "y_r10_g06_3000k",
                "y_r10_g07_3000k",
                "y_r10_g08_3000k",
                "y_r10_g09_3000k",
                "y_r10_g10_3000k"]
    
    
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    print(f"State shape: {state.shape}")
    # create a copy of state with random values between 0 and 1
    random_state = torch.rand_like(state, device=agent.device)
    print(f"Random state shape: {random_state.shape}")
    """
    #state_image = state[0].permute(2, 1, 0).cpu().numpy()
    state_image = random_state[0].permute(2, 1, 0).cpu().numpy()
    plt.imshow(state_image, cmap=None)
    plt.title("Random State")
    plt.colorbar()
    plt.show()
    """

    agent.load_model_weights(f"C:/Projects/public/XAI_Master/models/{modelNames[0]}.pth")

    def createRandomDataset(num_samples=100, agent=agent, target=0):
        """
        Create a random dataset of images and labels
        """
        # Create random images and labels
        states = np.random.rand(num_samples, 7, 7, 3)
        return
    

    
"""
create a random pictures
load models and get statistics
create one row with input and shap
illustrate
"""