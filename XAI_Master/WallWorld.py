import numpy as np
import pygame
import random
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces


class WallWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, size=6, agentSpawn = None, stepLoss=-0.01, maxSteps=100, wallCoordinates=None, randomWalls=None, redChestCoordinates=None, greenChestCoordinates=None, keyCoordinates=None, redChestReward=0.1, greenChestReward=0.1, randomredChests=None, randomgreenChests=None, randomkeys=None, saveImages=False, killSwitchSteps=6, agentSpawnCoordinates=None, chestSpawnCoordinates=None):
        self.size = size  # The size of the square grid
        self.agentSpawn = agentSpawn
        self.window_size = 512  # The size of the PyGame window
        self.q_values = None #np.zeros((self.size, self.size)) # Q-values for each cell and action
        self.q_value_actions = np.zeros((self.size, self.size))
        self.stepLoss = stepLoss
        self.maxSteps = maxSteps
        self.steps = 0
        self.initWallCoordinates = np.empty((0, 2), dtype=int) if wallCoordinates is None else wallCoordinates
        self.randomWalls = randomWalls
        self.wallCoordinates = self.initWallCoordinates.copy()
        self.initredChestCoordinates = np.empty((0, 2), dtype=int) if redChestCoordinates is None else redChestCoordinates
        self.initgreenChestCoordinates = np.empty((0, 2), dtype=int) if greenChestCoordinates is None else greenChestCoordinates
        self.initkeyCoordinates = np.empty((0, 2), dtype=int) if keyCoordinates is None else keyCoordinates
        self.redChestReward = redChestReward
        self.greenChestReward = greenChestReward
        self.randomredChests = randomredChests
        self.randomgreenChests = randomgreenChests
        self.randomkeys = randomkeys
        self.agentKeys = 0
        self.saveImages = saveImages
        self.killSwitchSteps = killSwitchSteps
        self.killSwitchCoordinates = np.empty((0, 2), dtype=int)
        self.killSwitchCount = 0
        self.agentSpawnCoordinates = agentSpawnCoordinates
        self.chestSpawnCoordinates = chestSpawnCoordinates

        self.wallColor = (0, 0, 0)
        self.blankColor = (255, 255, 255)
        self.agentColor = (0, 0, 255)
        self.redChestColor = (255, 0, 0)
        self.greenChestColor = (0, 255, 0)
        self.keyColor = (0, 255, 0)
        self.agentKeyColor = (0, 50, 0)
        self.q_valueMaxColor = (200, 135, 100)
        self.q_valueMinColor = (255, 230, 200)

        # Set the observation space as a matrix of size x size x 3 (for RGB)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.size, self.size, 1), dtype=np.uint8
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1, 0]), # Right
            1: np.array([0, 1]), # Down
            2: np.array([-1, 0]), # Left
            3: np.array([0, -1])} # Up
        
        self.action_to_arrow = {
            0: "🠪", # Right   """🠤🠥🠦🠧🠨🠩🠪🠫🠬"""
            1: "🠫", # Down
            2: "🠨", # Left
            3: "🠩"} # Up 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self, agentLoc=None):
        obs = np.full((self.size, self.size, 3), 255, dtype=np.float16)
        if agentLoc is None:
            agentLoc = self.agentCoordinates
        if self.redChestCoordinates is not None:
            for redChest in self.redChestCoordinates:
                obs[redChest[0], redChest[1]] = self.redChestColor
        if self.greenChestCoordinates is not None:
            for greenChest in self.greenChestCoordinates:
                obs[greenChest[0], greenChest[1]] = self.greenChestColor
        if self.keyCoordinates is not None:
            for key in self.keyCoordinates:
                obs[key[0], key[1]] = self.keyColor
        
        obs[agentLoc[0], agentLoc[1]] = self.agentColor
        obs[agentLoc[0], agentLoc[1]][1] = min(255, self.agentKeyColor[1]*self.agentKeys)

        if self.wallCoordinates is not None:
            for coordinate in self.wallCoordinates:
                obs[coordinate[0], coordinate[1]] = self.wallColor

        obs = obs / 255 # normalise
        return obs
    
    def _get_info(self):
        return None
    
    def isLocationOccupied(self, location):
        if np.any([np.array_equal(location, loc) for loc in self.wallCoordinates]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self.redChestCoordinates]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self.greenChestCoordinates]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self.keyCoordinates]):
            return True
        if np.array_equal(location, self.agentCoordinates):
            return True
        return False

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.steps = 0
        self.agentKeys = 0
        self.wallCoordinates = self.initWallCoordinates.copy()
        self.redChestCoordinates = self.initredChestCoordinates.copy()
        self.greenChestCoordinates = self.initgreenChestCoordinates.copy()
        self.keyCoordinates = self.initkeyCoordinates.copy()
        self.killSwitchCoordinates = np.empty((0, 2), dtype=int)
        self.killSwitchCount = 0

        if self.randomWalls is not None:
            for i in range(self.randomWalls):
                newRandWall = self.np_random.integers(0, self.size, size=2, dtype=int)
                while np.any([np.array_equal(newRandWall, wall) for wall in self.wallCoordinates]):
                    newRandWall = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.wallCoordinates = np.append(self.wallCoordinates, [newRandWall], axis=0)
        
        if self.agentSpawn is None:
            # Choose the locations uniformly random until they don't coincide with eachother or walls
            if self.agentSpawnCoordinates is not None:
                self.agentCoordinates = self.agentSpawnCoordinates[self.np_random.integers(0, len(self.agentSpawnCoordinates))]
            else:
                self.agentCoordinates = self.np_random.integers(0, self.size, size=2, dtype=int)
                if self.wallCoordinates is not None:
                    while np.any([np.array_equal(self.agentCoordinates, wall) for wall in self.wallCoordinates]):
                        self.agentCoordinates = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self.agentCoordinates = self.agentSpawn

        if self.randomredChests is not None:
            for i in range(self.randomredChests):
                if self.chestSpawnCoordinates is not None:
                    newRandredChest = self.chestSpawnCoordinates[self.np_random.integers(0, len(self.chestSpawnCoordinates))]
                    while self.isLocationOccupied(newRandredChest):
                        newRandredChest = self.chestSpawnCoordinates[self.np_random.integers(0, len(self.chestSpawnCoordinates))]
                else:
                    newRandredChest = self.np_random.integers(0, self.size, size=2, dtype=int)
                    while self.isLocationOccupied(newRandredChest):
                        newRandredChest = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.redChestCoordinates = np.append(self.redChestCoordinates, [newRandredChest], axis=0)

        if self.randomgreenChests is not None:
            for i in range(self.randomgreenChests):
                if self.chestSpawnCoordinates is not None:
                    newRandgreenChest = self.chestSpawnCoordinates[self.np_random.integers(0, len(self.chestSpawnCoordinates))]
                    while self.isLocationOccupied(newRandgreenChest):
                        newRandgreenChest = self.chestSpawnCoordinates[self.np_random.integers(0, len(self.chestSpawnCoordinates))]
                else:
                    newRandgreenChest = self.np_random.integers(0, self.size, size=2, dtype=int)
                    while self.isLocationOccupied(newRandgreenChest):
                        newRandgreenChest = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.greenChestCoordinates = np.append(self.greenChestCoordinates, [newRandgreenChest], axis=0)

        if self.randomkeys is not None:
            for i in range(self.randomkeys):
                newRandkey = self.np_random.integers(0, self.size, size=2, dtype=int)
                while self.isLocationOccupied(newRandkey):
                    newRandkey = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.keyCoordinates = np.append(self.keyCoordinates, [newRandkey], axis=0)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Map the action {0,1,2,3} to direction right, down, left, up
        direction = self.action_to_direction[action]

        # Check if the agent is trying to walk into a wall
        if np.any([np.array_equal(self.agentCoordinates + direction, wall) for wall in self.wallCoordinates]):
            direction = np.array([0, 0])
        self.agentCoordinates = np.clip(self.agentCoordinates + direction, 0, self.size - 1) # Make sure the agent doesn't leave the grid

        terminated = False
        reward = self.stepLoss

        # I want to implement a kill switch that checks if the agent has been in the same location for a certain number of steps
        # The kill switch should have up to 2 coordinates that are updated fifo every step
        # If the agent has been in the same location for a killSwitchSteps number of steps, then terminated=true
        if len(self.killSwitchCoordinates) < 2:
            self.killSwitchCoordinates = np.append(self.killSwitchCoordinates, [self.agentCoordinates], axis=0)
        elif np.any([np.array_equal(self.agentCoordinates, loc) for loc in self.killSwitchCoordinates]):
            self.killSwitchCount += 1
            if self.killSwitchCount > 2:
                if self.render_mode == "human":
                    print(f"{self.killSwitchCount}")
        else:
            self.killSwitchCount = 0
            self.killSwitchCoordinates = np.delete(self.killSwitchCoordinates, 0, axis=0)
            self.killSwitchCoordinates = np.append(self.killSwitchCoordinates, [self.agentCoordinates], axis=0)
        truncated = self.killSwitchCount >= self.killSwitchSteps
        if self.killSwitchCount >= self.killSwitchSteps:
            if self.render_mode == "human":
                print(f"Trunc!!")

        if np.any([np.array_equal(self.agentCoordinates, loc) for loc in self.redChestCoordinates]):
            reward = self.redChestReward
            self.redChestCoordinates = np.array([loc for loc in self.redChestCoordinates if not np.array_equal(self.agentCoordinates, loc)])
            if self.redChestCoordinates.size == 0:
                terminated = True
        
        if np.any([np.array_equal(self.agentCoordinates, loc) for loc in self.greenChestCoordinates]):
            reward = self.greenChestReward
            self.greenChestCoordinates = np.array([loc for loc in self.greenChestCoordinates if not np.array_equal(self.agentCoordinates, loc)])
            if self.greenChestCoordinates.size == 0:
                terminated = True
        
        elif np.any([np.array_equal(self.agentCoordinates, loc) for loc in self.keyCoordinates]):
            self.agentKeys += 1
            self.keyCoordinates = np.array([loc for loc in self.keyCoordinates if not np.array_equal(self.agentCoordinates, loc)])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.steps += 1
        truncated = truncated or self.steps >= self.maxSteps

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size # The size of a single grid square in pixels

        # Display Q-values as colored cell background
        if self.q_values is not None:
            for i, j in np.ndindex(self.q_values.shape):
                pygame.draw.rect(
                    canvas,
                    tuple(int(self.q_valueMaxColor[c] * self.q_values[i, j] + self.q_valueMinColor[c] * (1 - self.q_values[i, j]))for c in range(3)),  # Iterate over R, G, B channels
                    pygame.Rect(
                        pix_square_size * np.array([i, j]),  # Position of the cell
                        (pix_square_size, pix_square_size),  # Size of the cell
                    ),
                )
    
        # Display q-value arrows
        if self.q_value_actions is not None:
            for i, j in np.ndindex(self.q_value_actions.shape):
                font = pygame.font.SysFont('Segoe UI Symbol', int(pix_square_size/ 1.7))
                text = font.render(self.action_to_arrow[self.q_value_actions[i, j]], True, (255, 255, 255))
                text_rect = text.get_rect(center=(pix_square_size * i + pix_square_size / 2, pix_square_size * j + pix_square_size / 2))
                canvas.blit(text, text_rect)

        # Display walls
        if self.wallCoordinates is not None:
            for wall in self.wallCoordinates:
                pygame.draw.rect(
                    canvas,
                    self.wallColor,
                    pygame.Rect(
                        pix_square_size * wall,  # Position of the cell
                        (pix_square_size, pix_square_size),  # Size of the cell
                    ),
                )
        
        # Display agents keys
        agentKeyColor = (max(0, self.blankColor[0]-self.agentKeyColor[1]*self.agentKeys), self.blankColor[1], max(0, self.blankColor[2]-self.agentKeyColor[1]*self.agentKeys))
        pygame.draw.circle(
            canvas,
            agentKeyColor,
            (self.agentCoordinates + 0.5) * pix_square_size,
            pix_square_size / 2,
        )
        # Display agent
        pygame.draw.circle(
            canvas,
            self.agentColor,
            (self.agentCoordinates + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Display redChests
        for redChest in self.redChestCoordinates:
            pygame.draw.circle(
                canvas,
                self.redChestColor,
                (redChest + 0.5) * pix_square_size,
                pix_square_size / 4,
            )
        # Display greenChests
        for greenChest in self.greenChestCoordinates:
            pygame.draw.circle(
                canvas,
                self.greenChestColor,
                (greenChest + 0.5) * pix_square_size,
                pix_square_size / 4,
            )
        # Display keys
        for key in self.keyCoordinates:
            pygame.draw.circle(
                canvas,
                self.keyColor,
                (key + 0.5) * pix_square_size,
                pix_square_size / 4,
            )

        # Add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.saveImages:
            frame_number = getattr(self, '_frame_number', 0)  # Keep track of frame numbers
            pygame.image.save(canvas, f"frame_{frame_number:04d}.png")
            print(f"Saved frame_{frame_number:04d}.png")
            self._frame_number = frame_number + 1  # Increment frame counter

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# if main

if __name__ == "__main__":

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
    env = WallWorld(render_mode="human", size=8, randomredChests=1, randomgreenChests=1, randomkeys=2, randomWalls=5)
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
