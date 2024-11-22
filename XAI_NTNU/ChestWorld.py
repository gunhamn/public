import numpy as np
import pygame
import random
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces


class ChestWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None, size=6, agentSpawn = None, stepLoss=-0.01, maxSteps=100, wallCoordinates=None, randomWalls=None, chestCoordinates=None, keyCoordinates=None, chestReward=0.1, randomchests=None, randomkeys=None):
        self.size = size  # The size of the square grid
        self.agentSpawn = agentSpawn
        self.window_size = 512  # The size of the PyGame window
        self.q_values = np.zeros((self.size, self.size)) # Q-values for each cell and action
        self.stepLoss = stepLoss
        self.maxSteps = maxSteps
        self.steps = 0
        self.initWallCoordinates = np.empty((0, 2), dtype=int) if wallCoordinates is None else wallCoordinates
        self.randomWalls = randomWalls
        self.wallCoordinates = self.initWallCoordinates.copy()
        self.initchestCoordinates = np.empty((0, 2), dtype=int) if chestCoordinates is None else chestCoordinates
        self.initkeyCoordinates = np.empty((0, 2), dtype=int) if keyCoordinates is None else keyCoordinates
        self.chestReward = chestReward
        self.randomchests = randomchests
        self.randomkeys = randomkeys
        self.agentKeys = 0

        self.wallColor = (0, 0, 0)
        self.blankColor = (255, 255, 255)
        self.agentColor = (0, 0, 255)
        self.chestColor = (255, 0, 0)
        self.keyColor = (0, 255, 0)
        self.agentKeyColor = (0, 50, 0)
        self.q_valueMaxColor = (255, 100, 100)
        self.q_valueMinColor = (255, 255, 255)

        # Set the observation space as a matrix of size x size x 3 (for RGB)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.size, self.size, 1), dtype=np.uint8
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]), # Right
            1: np.array([0, 1]), # Down
            2: np.array([-1, 0]), # Left
            3: np.array([0, -1])} # Up

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
            agentLoc = self._agent_location
        if self.chestCoordinates is not None:
            for chest in self.chestCoordinates:
                obs[chest[0], chest[1]] = self.chestColor
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
        if np.any([np.array_equal(location, loc) for loc in self.chestCoordinates]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self.keyCoordinates]):
            return True
        if np.array_equal(location, self._agent_location):
            return True
        return False

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.steps = 0
        self.agentKeys = 0
        self.wallCoordinates = self.initWallCoordinates.copy()
        self.chestCoordinates = self.initchestCoordinates.copy()
        self.keyCoordinates = self.initkeyCoordinates.copy()

        if self.randomWalls is not None:
            for i in range(self.randomWalls):
                newRandWall = self.np_random.integers(0, self.size, size=2, dtype=int)
                while np.any([np.array_equal(newRandWall, wall) for wall in self.wallCoordinates]):
                    newRandWall = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.wallCoordinates = np.append(self.wallCoordinates, [newRandWall], axis=0)
        
        if self.agentSpawn is None:
            # Choose the locations uniformly random until they don't coincide with eachother or walls
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if self.wallCoordinates is not None:
                while np.any([np.array_equal(self._agent_location, wall) for wall in self.wallCoordinates]):
                    self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self._agent_location = self.agentSpawn

        if self.randomchests is not None:
            for i in range(self.randomchests):
                newRandchest = self.np_random.integers(0, self.size, size=2, dtype=int)
                while self.isLocationOccupied(newRandchest):
                    newRandchest = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.chestCoordinates = np.append(self.chestCoordinates, [newRandchest], axis=0)

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
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Check if the agent is trying to walk into a wall
        if np.any([np.array_equal(self._agent_location + direction, wall) for wall in self.wallCoordinates]):
            direction = np.array([0, 0])
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1) # Make sure the agent doesn't leave the grid

        terminated = False
        reward = self.stepLoss

        if np.any([np.array_equal(self._agent_location, loc) for loc in self.chestCoordinates]):
            if self.agentKeys > 0:
                reward = self.chestReward
                self.chestCoordinates = np.array([loc for loc in self.chestCoordinates if not np.array_equal(self._agent_location, loc)])
                self.agentKeys -= 1
                if self.chestCoordinates.size == 0 or (self.keyCoordinates.size == 0 and self.agentKeys == 0):
                    terminated = True
        
        elif np.any([np.array_equal(self._agent_location, loc) for loc in self.keyCoordinates]):
            self.agentKeys += 1
            self.keyCoordinates = np.array([loc for loc in self.keyCoordinates if not np.array_equal(self._agent_location, loc)])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.steps += 1
        truncated = self.steps >= self.maxSteps

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size # The size of a single grid square in pixels

        # Display Q-values as colored cell background
        for i, j in np.ndindex(self.q_values.shape):
            pygame.draw.rect(
                canvas,
                tuple(int(self.q_valueMaxColor[c] * self.q_values[i, j] + self.q_valueMinColor[c] * (1 - self.q_values[i, j]))for c in range(3)),  # Iterate over R, G, B channels
                pygame.Rect(
                    pix_square_size * np.array([i, j]),  # Position of the cell
                    (pix_square_size, pix_square_size),  # Size of the cell
                ),
            )
    
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
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 2,
        )
        # Display agent
        pygame.draw.circle(
            canvas,
            self.agentColor,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Display chests
        for chest in self.chestCoordinates:
            pygame.draw.circle(
                canvas,
                self.chestColor,
                (chest + 0.5) * pix_square_size,
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
    env = ChestWorld(render_mode="human", size=8, randomchests=3, randomkeys=2, randomWalls=5)
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
