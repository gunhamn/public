import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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
    redChestReward=-1
    greenChestReward=1
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
    
    maxSteps = 30
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
    
    agent.load_model_weights(f"C:/Projects/public/XAI_Master/models/WW_redReward1_grReward0_7x7_700000steps.pth")
    #agent.inference(env=show_env, max_steps=100, epsilon=epsilon_min, renderQvalues=False)
    
    # First define the hook and dictionary to store activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hook on the last layer (fc2)
    hook = agent.policy_net.fc1.register_forward_hook(get_activation('fc1'))

    # Your existing code
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    action = agent.policy_net(state).max(1)[1].view(1, 1)
    redChestCoordinates = env.redChestCoordinates
    greenChestCoordinates = env.greenChestCoordinates
    agentLoc = env.agentCoordinates

    # Print the activations (this will show
    # all values before the max operation)
    print("Last layer activations:", activations['fc1'])

    activationData = activations['fc1'].cpu().numpy()[0].flatten()
    stateData = state.cpu().numpy()[0].flatten()

    print(f"Activation data: {activationData}")
    print(f"State data: {stateData}")
    # print len
    print(f"Activation data len: {len(activationData)}")
    print(f"State data len: {len(stateData)}")
    print(f"Red chest coordinates: {redChestCoordinates}")
    print(f"Red chest x coordinate: {redChestCoordinates[0][0]}")
    print(f"Red chest y coordinate: {redChestCoordinates[0][1]}")
    
    print(f"Green chest coordinates: {greenChestCoordinates}")
    print(f"Agent coordinates: {agentLoc}")

    # Remove the hook when finished
    hook.remove()
    print(f"Action: {action}")
    print(agent.policy_net)

    print("Create dataset")
    df = agent.createDataset(env, num_episodes=5000)
    print(df)

    df.to_csv(f"C:/Projects/public/XAI_Master/datasets/red1green0.csv", 
          index=False,          # No index as column
          float_format='%.8f'   # Round to 8 decimals
         )


"""Todo for tomorrow:
Create a complete create dataset function
where each row contains the state, activations and result.
Since there are almost 200 features, 2 training sets with
5000 samples each should be created, then mixed, then saving
5% for testing. Then a ipynb file should be created that uses
these csv files to predict outcome."""