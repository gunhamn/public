import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap
import time

from WallWorld import WallWorld
from DqnAgent import DqnAgent

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
                    chestSpawnCoordinates=chestSpawnCoordinates)
    observation, _ = env.reset()
    agent = DqnAgent(env.action_space, observation,
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
                    chestSpawnCoordinates=chestSpawnCoordinates)

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
    
    gamestateDataset = pd.read_csv("C:/Projects/public/XAI_Master/datasets/42000_gamestates.csv")

    def plot_image(row):
        image_values = row.values.reshape(7, 7, 3).transpose((1, 0, 2))
        plt.imshow(image_values)
        plt.suptitle(f'Game state')
        plt.axis('off')
        plt.show()
    # plot_image(gamestateDataset.iloc[0])

    def createShapBackgroundDataset(pixelDataset, sampleSize=1000):
        if pixelDataset.shape[0] > sampleSize:
            backgroundData = pixelDataset.sample(sampleSize)
        else:
            backgroundData = pixelDataset
        
        backgroundData = backgroundData.values.reshape(-1, 7, 7, 3)#.transpose((0, 2, 1, 3))
        backgroundData = torch.tensor(backgroundData, dtype=torch.float32)
        return backgroundData
    
    background_data_40k_samples = createShapBackgroundDataset(gamestateDataset, sampleSize=40000)
    print(background_data_40k_samples.shape)
    
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    backgroundData = agent.createUncoloredState(state)
    print(backgroundData.shape)
    
    agent.load_model_weights(f"C:/Projects/public/XAI_Master/models/r10_g10_1500k.pth")
    df = agent.createShapDataset(env, backgroundData=background_data_40k_samples, batch_size=40000, num_episodes=10)
    df.to_csv(f"C:/Projects/public/XAI_Master/datasets/shap_test.csv", 
                index=False)# No index as column
                # float_format='%.8f')   # Round to 8 decimals
    print("Complete")
    
    
    timeStart = time.time()
    for modelName, percent in zip(modelNames, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                               0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]):
        print(f"Creating dataset for {modelName}")
        agent.load_model_weights(f"C:/Projects/public/XAI_Master/models/{modelName}.pth")
        df = agent.createShapDataset(env, backgroundData=background_data_40k_samples, batch_size=40000, num_episodes=2000)
        df.to_csv(f"C:/Projects/public/XAI_Master/datasets/shap_{modelName}.csv", 
                index=False)# No index as column
                # float_format='%.8f')   # Round to 8 decimals
        """
        df = agent.createActivationDataset(env, num_episodes=2000)
        df.to_csv(f"C:/Projects/public/XAI_Master/datasets/act_{modelName}.csv", 
                index=False)# No index as column
                # float_format='%.8f')   # Round to 8 decimals
        """
        agent.printProgress(timeStart, percent=percent)
    print("Complete") # Exp: 5h runtime to shap all 21aaz
    
