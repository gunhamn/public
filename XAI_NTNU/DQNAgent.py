import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]  # Flatten input

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get dimensions of state and action space
        input_shape = env.observation_space.shape
        output_dim = env.action_space.n

        # Initialize DQN and target network
        self.dqn = DQN(input_shape, output_dim).to(self.device)
        self.target_dqn = DQN(input_shape, output_dim).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())  # Initialize target network with same weights

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        self.action_direction = {
            0: "→",
            1: "↓",
            2: "←",
            3: "↑"}

    def get_action(self, state, printChoices=False):
        if np.random.rand() < self.epsilon:
            if printChoices:
                print("Random choice")
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.dqn(state)
        if printChoices:
            print(f"Q-values: {q_values}")
        return q_values.argmax().item()

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a batch from the replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Calculate target Q-values
        with torch.no_grad():
            target_q_values = reward + self.gamma * self.target_dqn(next_state).max(1)[0] * (1 - done)

        # Get the current Q-values for the actions taken
        current_q_values = self.dqn(state).gather(1, action).squeeze(1)

        # Compute the loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, env=None, ep_length=100, printChoices=False, plot=False):
        if env is None:
            env = self.env
        
        rewards = []  # List to store total rewards for each episode
        floating_avg = []  # List to store 10-episode floating average of rewards
        
        if plot:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            line, = ax.plot([], [], label="Reward per Ep")
            line2, = ax.plot([], [], label="10-Ep Floating Avg")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Training Progress")
            plt.legend(loc="lower left")


        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while steps < ep_length and not done:
                action = self.get_action(state, printChoices=printChoices)
                if printChoices:
                    print(f"Target: {env._target_location} agent: {env._agent_location} action: {action} {self.action_direction[action]}")
                next_state, reward, done, _, _ = env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

                self.update()

            # Store total reward for this episode
            rewards.append(total_reward)
            floating_avg.append(np.mean(rewards[-10:]))

            # Update the plot dynamically
            if plot:
                line.set_xdata(range(len(rewards)))
                line.set_ydata(rewards)
                line2.set_xdata(range(len(floating_avg)))
                line2.set_ydata(floating_avg)
                ax.relim()  # Recalculate limits to fit new data
                ax.autoscale_view()  # Rescale the view to fit new data
                plt.draw()
                plt.pause(0.01)  # Pause to update the plot

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Update target network periodically
            if episode % 10 == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

            print(f"Episode {episode}, Total Reward: {total_reward}")

        if plot:
            plt.ioff()  # Turn off interactive mode after training is done
            plt.show()  # Show final plot


    def getQValues(self, env):
        q_values = np.zeros((env.size, env.size, 2), dtype=object)
        for i in range(env.size):
            for j in range(env.size):
                obs = env._get_obs(agentLoc=(i, j))
                state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values[i, j, 0] = round(self.dqn(state).max().item(), 2)
                q_values[i, j, 1] = self.action_direction[self.dqn(state).argmax().item()]
        q_values[env._target_location[0], env._target_location[1], 1] = "⬤"
        return q_values
    
    def getBestDirection(self, env):
        bestDirections = np.zeros((env.size, env.size), dtype=object)
        for i in range(env.size):
            for j in range(env.size):
                obs = env._get_obs(agentLoc=(i, j))
                state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                bestDirections[i, j] = self.action_direction[self.dqn(state).argmax().item()]
        bestDirections[env._target_location[0], env._target_location[1]] = "⬤"
        print(f"Fictive_state, Target: {env._target_location} agent: [{env._target_location[0]} {env._target_location[1]-1}]")
        fictive_obs = env._get_obs(agentLoc=(env._target_location[0], env._target_location[0]-1))
        fictive_state = torch.FloatTensor(fictive_obs).unsqueeze(0).to(self.device)
        print(f"q values: {self.dqn(fictive_state)} action: {self.dqn(fictive_state).argmax().item()} {self.action_direction[self.dqn(fictive_state).argmax().item()]}")
        return bestDirections.transpose()

    def save_model(self, file_path):
        """Save the DQN model to a file."""
        torch.save({
            'model_state_dict': self.dqn.state_dict(),
            'target_model_state_dict': self.target_dqn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,  # Save epsilon for continued training
        }, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the DQN model from a file."""
        checkpoint = torch.load(file_path)
        self.dqn.load_state_dict(checkpoint['model_state_dict'])
        self.target_dqn.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']  # Load epsilon for continued training
        print(f"Model loaded from {file_path}")
    
if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env_rendered = gym.make('CartPole-v1', render_mode='human')
    agent = DQNAgent(env)

    # Train for 500 episodes
    agent.train(100)
    agent.train(1, env=env_rendered)
    agent.train(100)
    agent.train(1, env=env_rendered)
    agent.train(100)
    agent.train(1, env=env_rendered)
    agent.train(100)
    agent.train(1, env=env_rendered)
    agent.train(100)
    agent.train(1, env=env_rendered)
    agent.save_model('dqn_cartpole.pth')
