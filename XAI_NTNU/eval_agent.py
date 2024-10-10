from DQNGridWorldEnv import DQNGridWorldEnv
from dqn_agent import DQNAgent
import numpy as np
import shap


# Define a function to predict Q-values
def predict_q_values(agent, observations):
    q_values = []
    for obs in observations:
        q_values.append(agent.model.predict(np.array([obs])))
    return np.array(q_values)

if __name__ == "__main__":

    # Config
    num_episodes = 50

    # DQNGridWorldEnv
    size=5
    agentSpawn=None
    targetSpawn=None
    goalReward=0.5
    stepLoss=-0.01
    maxSteps=50
    wallCoordinates=np.array([[1, 1], [1, 3], [3, 1], [3, 3]])
    forbiddenCoordinates=np.array([[2, 2]])
    forbiddenPenalty=-0.5
    chanceOfSupervisor=0.5
    randomWalls=0
    randomForbiddens=0

    # Agent
    useWandb = False
    batch_size=128
    lr=0.0001
    gamma=0.99
    epsilon_start=0.9
    epsilon_min=0.05
    epsilon_decay=20_000
    tau=0.005

    env = DQNGridWorldEnv(render_mode="human", size=size, agentSpawn=agentSpawn, targetSpawn=targetSpawn, goalReward=goalReward, stepLoss=stepLoss, maxSteps=maxSteps, wallCoordinates=wallCoordinates, forbiddenCoordinates=forbiddenCoordinates, forbiddenPenalty=forbiddenPenalty, chanceOfSupervisor=chanceOfSupervisor, randomWalls=randomWalls, randomForbiddens=randomForbiddens)
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
    
    agent.load_model_weights("C:/Projects/public/XAI_NTNU/models/X-absSup_20perc_5x5_30000ep.pth")
    print(f"Observation: {observation}")
    agent.inference(env, num_episodes)
    #observation, _ = env.reset()
    #agent.get_q_values(observation)



    """
    # Generate some observations
    observations = [env.reset() for _ in range(100)]

    # Create a SHAP explainer
    explainer = shap.KernelExplainer(lambda x: predict_q_values(agent, x), observations[:50])

    # Calculate SHAP values for a single observation
    shap_values = explainer.shap_values(observations[0:1])

    # Plot the SHAP values
    shap.summary_plot(shap_values, observations[0:1])
    """