import gymnasium as gym
import pygame
from gridWorldEnv import GridWorldEnv

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
    pygame.K_UP: 3    # Up
}

# Initialize the environment
env = GridWorldEnv(render_mode="human", size=3, randomSpawn=False)
observation, info = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_action_mapping:
                action = key_action_mapping[event.key]
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    observation, info = env.reset()

    env.render()  # Update the display with the new state

# Clean up
env.close()
pygame.quit()