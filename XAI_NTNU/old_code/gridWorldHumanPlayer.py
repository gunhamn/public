import gymnasium as gym
import pygame
from DQNGridWorldEnv import DQNGridWorldEnv
import numpy as np

# Initialize pygame
pygame.init()

# Set up display
window_size = 500  # Change this if your environment's render size is different
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption('GridWorld Manual Control')

# Define key mappings for actions
key_action_mapping = {
    pygame.K_RIGHT: 0, # Right
    pygame.K_DOWN: 1,  # Down
    pygame.K_LEFT: 2,   # Left
    pygame.K_UP: 3}    # Up

# Initialize the environment
# env = DQNGridWorldEnv(render_mode="human", size=8, wallCoordinates=np.array([[1, 1], [1, 2], [1, 3], [2, 4], [3, 4], [4, 2], [4, 3], [4, 4], [5, 5], [6, 6]]))
env = DQNGridWorldEnv(render_mode="human")
observation, info = env.reset()
print(f"Observation: {observation}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_action_mapping:
                action = key_action_mapping[event.key]
                observation, reward, terminated, truncated, info = env.step(action)
                print(f"Observation: {observation}")
                print(f"Reward: {reward}")
                if terminated or truncated:
                    observation, info = env.reset()

    env.render()  # Update the display with the new state

# Clean up
env.close()
pygame.quit()
