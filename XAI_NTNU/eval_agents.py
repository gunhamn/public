
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
from dqn_agent_new import DqnAgentNew



if __name__ == "__main__":

    # Config
    max_steps=1_000_000

    # ChestWorld
    render_mode=None
    size=6
    agentSpawn = None
    q_values=None
    maxSteps=200
    stepLoss=-1/maxSteps # min reward should be -1
    wallCoordinates=None
    randomWalls=2
    chestCoordinates=None
    keyCoordinates=None
    randomchests=3
    randomkeys=6
    chestReward=1/min(randomchests, randomkeys) # max reward should be 1
    
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

    useWandb = True
    saveModel = True

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

    env = ChestWorld(render_mode=None, size=size, agentSpawn=agentSpawn, q_values=q_values, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates, randomWalls=randomWalls, chestCoordinates=chestCoordinates, keyCoordinates=keyCoordinates, chestReward=chestReward, randomchests=randomchests, randomkeys=randomkeys)
    observation, _ = env.reset()
    agent = DqnAgentNew(env.action_space, observation,
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
    agent.load_model_weights(f"C:/Projects/public/XAI_NTNU/models/2goodCoins0walls8x8_3000ep.pth")
    
    maxSteps = 30
    show_env = ChestWorld(render_mode="human", size=size, agentSpawn=agentSpawn, q_values=q_values, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates, randomWalls=randomWalls, chestCoordinates=chestCoordinates, keyCoordinates=keyCoordinates, chestReward=chestReward, randomchests=randomchests, randomkeys=randomkeys)
    agent.wandb = False
    agent.inference(env=show_env, max_steps=max_steps, epsilon=epsilon_min)