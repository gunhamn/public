from DQNGridWorldEnv import DQNGridWorldEnv
from dqn_agent import DQNAgent
import numpy as np



if __name__ == "__main__":

    # Config
    num_episodes = 20

    # DQNGridWorldEnv
    size=8
    goalReward=2
    stepLoss=-0.01
    maxSteps=20
    wallCoordinates = None
    # wallCoordinates=np.array([[1, 1], [1, 2], [1, 3], [2, 3], [3, 3]])

    # Agent
    batch_size=128
    lr=0.0001
    gamma=0.99
    epsilon_start=0.9
    epsilon_min=0.05
    epsilon_decay=100_000
    tau=0.005


    env = DQNGridWorldEnv(render_mode="human", size=size, goalReward=goalReward, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates)
    observation, info = env.reset()
    agent = DQNAgent(env.action_space, observation,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        tau=tau,
        wandb=None)
    
    agent.load_model_weights("C:/Projects/public/XAI_NTNU/models/8x8_12000ep.pth")
    print(f"Observation: {observation}")
    agent.inference(env, num_episodes)