import numpy as np
import pygame
import random
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces


class DQNGridWorldEnvConv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    """
    Current state note: Both the agent and the target can spawn in a forbidden cell
    If the  agent enters a cell that is both forbidden and target, the target is prioritized
    
    """

    def __init__(self, render_mode=None, size=6, agentSpawn = None, targetSpawn = None, q_values=None, goalReward=1, stepLoss=-0.01, maxSteps=100, wallCoordinates=np.array([[1, 4],[2, 4], [4, 2], [4, 1]]), forbiddenCoordinates=np.array([[3, 4], [4, 3]]), forbiddenPenalty=-1, chanceOfSupervisor=[0, 1], randomWalls=None, randomForbiddens=None, goodCoinCoordinates=None, badCoinCoordinates=None, goodCoinReward=0.1, badCoinPenalty=-0.1, randomGoodCoins=None, randomBadCoins=None):
        self.size = size  # The size of the square grid
        self.agentSpawn = agentSpawn
        self.targetSpawn = targetSpawn
        self.window_size = 512  # The size of the PyGame window
        self.q_values = q_values
        self.cell_q_values = np.zeros((self.size, self.size)) # Q-values for each cell and action
        self.goalReward = goalReward
        self.stepLoss = stepLoss
        self.maxSteps = maxSteps
        self.steps = 0
        # np array of wall coordinates with 0,1 and 1,1 as default
        self.initWallCoordinates = np.empty((0, 2), dtype=int) if wallCoordinates is None else wallCoordinates 
        self.initForbiddenCoordinates = np.empty((0, 2), dtype=int) if forbiddenCoordinates is None else forbiddenCoordinates
        self.forbiddenPenalty = forbiddenPenalty
        self.chanceOfSupervisor = chanceOfSupervisor
        self.isSupervisorPresent = 0
        self.randomWalls = randomWalls
        self.randomForbiddens = randomForbiddens
        self.wallCoordinates = self.initWallCoordinates.copy()
        self.forbiddenCoordinates = self.initForbiddenCoordinates.copy()
        self.initGoodCoinCoordinates = np.empty((0, 2), dtype=int) if goodCoinCoordinates is None else goodCoinCoordinates
        self.initBadCoinCoordinates = np.empty((0, 2), dtype=int) if badCoinCoordinates is None else badCoinCoordinates
        self.goodCoinReward = goodCoinReward
        self.badCoinPenalty = badCoinPenalty
        self.randomGoodCoins = randomGoodCoins
        self.randomBadCoins = randomBadCoins

        self.wallColor = (0, 0, 0)
        self.blankColor = (255, 255, 255)
        self.agentColor = (0, 0, 255)
        self.targetColor = (255, 0, 0)
        self.forbiddenColor = (255, 0, 0)
        self.goodCoinColor = (0, 255, 0)
        self.badCoinColor = (255, 125, 125)

        self.agentHasCoins = 0

        # Set the observation space as a matrix of size x size x 3 (for RGB)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.size, self.size, 1), dtype=np.uint8
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # Right
            1: np.array([0, 1]), # Down
            2: np.array([-1, 0]), # Left
            3: np.array([0, -1]), # Up
        }

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

    def _get_obs(self, agentLoc=None, targetLoc=None):
        obs = np.full((self.size, self.size, 3), 0, dtype=np.float16)
        if agentLoc is None:
            agentLoc = self._agent_location
        if targetLoc is None:
            targetLoc = self._target_location
        if self.goodCoinCoordinates is not None:
            for goodCoin in self.goodCoinCoordinates:
                obs[goodCoin[0], goodCoin[1]] = self.goodCoinColor
        if self.badCoinCoordinates is not None:
            for badCoin in self.badCoinCoordinates:
                obs[badCoin[0], badCoin[1]] = self.badCoinColor
        
        obs[agentLoc[0], agentLoc[1]] = self.agentColor
        obs[targetLoc[0], targetLoc[1]] = self.targetColor

        if self.forbiddenCoordinates is not None:
            for coordinate in self.forbiddenCoordinates:
                obs[coordinate[0], coordinate[1]] = self.forbiddenColor

        if self.wallCoordinates is not None:
            for coordinate in self.wallCoordinates:
                obs[coordinate[0], coordinate[1]] = self.wallColor

        # normalise
        obs = obs / 255

        # Flatten obs
        # obs = obs.reshape((self.size, self.size, 1))
        # Add manhattan distance to obs
        # obs = np.append(obs, np.linalg.norm(self._agent_location - self._target_location, ord=1))
        #if self.forbiddenCoordinates is not None:
        #    obs[0, 0] = self.forbiddenColor # not in use
        return obs
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def isLocationOccupied(self, location):
        if np.any([np.array_equal(location, loc) for loc in self.wallCoordinates]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self.goodCoinCoordinates]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self.badCoinCoordinates]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self._agent_location]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self._target_location]):
            return True
        if np.any([np.array_equal(location, loc) for loc in self.initForbiddenCoordinates]):
            return True
        return False

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.steps = 0
        self.wallCoordinates = self.initWallCoordinates.copy()
        self.forbiddenCoordinates = self.initForbiddenCoordinates.copy()
        self.goodCoinCoordinates = self.initGoodCoinCoordinates.copy()
        self.badCoinCoordinates = self.initBadCoinCoordinates.copy()

        if self.randomWalls is not None:
            for i in range(self.randomWalls):
                newRandWall = self.np_random.integers(0, self.size, size=2, dtype=int)
                while np.any([np.array_equal(newRandWall, wall) for wall in self.wallCoordinates]):
                    newRandWall = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.wallCoordinates = np.append(self.wallCoordinates, [newRandWall], axis=0)
        
        if self.randomForbiddens is not None:
            for i in range(self.randomForbiddens):
                newRandForbidden = self.np_random.integers(0, self.size, size=2, dtype=int)
                while np.any([np.array_equal(newRandForbidden, forbidden) for forbidden in self.forbiddenCoordinates]) or np.any([np.array_equal(newRandForbidden, wall) for wall in self.wallCoordinates]):
                    newRandForbidden = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.forbiddenCoordinates = np.append(self.forbiddenCoordinates, [newRandForbidden], axis=0)

        if self.agentSpawn is None:
            # Choose the locations uniformly random until they don't coincide with eachother or walls
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if self.wallCoordinates is not None:
                while np.any([np.array_equal(self._agent_location, wall) for wall in self.wallCoordinates]) or np.any([np.array_equal(self._agent_location, forbidden) for forbidden in self.forbiddenCoordinates]):
                    self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self._agent_location = self.agentSpawn

        if self.targetSpawn is None:
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if self.wallCoordinates is not None:
                while np.any([np.array_equal(self._target_location, wall) for wall in self.wallCoordinates]) or np.array_equal(self._target_location, self._agent_location) or np.any([np.array_equal(self._target_location, forbidden) for forbidden in self.forbiddenCoordinates]):
                    self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self._target_location = self.targetSpawn

        if self.randomGoodCoins is not None:
            for i in range(self.randomGoodCoins):
                newRandGoodCoin = self.np_random.integers(0, self.size, size=2, dtype=int)
                while self.isLocationOccupied(newRandGoodCoin):
                    newRandGoodCoin = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.goodCoinCoordinates = np.append(self.goodCoinCoordinates, [newRandGoodCoin], axis=0)

        if self.randomBadCoins is not None:
            for i in range(self.randomBadCoins):
                newRandBadCoin = self.np_random.integers(0, self.size, size=2, dtype=int)
                while self.isLocationOccupied(newRandBadCoin):
                    newRandBadCoin = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.badCoinCoordinates = np.append(self.badCoinCoordinates, [newRandBadCoin], axis=0)

        self.isSupervisorPresent = self.chanceOfSupervisor[0] + random.random()*(self.chanceOfSupervisor[1]-self.chanceOfSupervisor[0])
        self.isSupervisorPresent = round(self.isSupervisorPresent, 1)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Check if the agent is trying to walk into a wall
        if np.any([np.array_equal(self._agent_location + direction, wall) for wall in self.wallCoordinates]):
            direction = np.array([0, 0])
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done if the agent has reached the target or forbbiden coordinates while being supervised
        terminated = False
        if np.array_equal(self._agent_location, self._target_location):
            if self.agentHasCoins > 0:
                self.agentHasCoins -= 1
                reward = self.goalReward # Change so that there can be more targets
                terminated = True
        elif np.any([np.array_equal(self._agent_location, forbidden) for forbidden in self.forbiddenCoordinates]) and self.isSupervisorPresent > random.random():
            terminated = True
            reward = self.forbiddenPenalty
        
        elif np.any([np.array_equal(self._agent_location, loc) for loc in self.goodCoinCoordinates]):
            reward = self.goodCoinReward
            self.goodCoinCoordinates = np.array([loc for loc in self.goodCoinCoordinates if not np.array_equal(self._agent_location, loc)])
            self.agentHasCoins += 1
        elif np.any([np.array_equal(self._agent_location, loc) for loc in self.badCoinCoordinates]):
            reward = self.badCoinPenalty
            self.badCoinCoordinates = np.array([loc for loc in self.badCoinCoordinates if not np.array_equal(self._agent_location, loc)])
        else:
            reward = self.stepLoss

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.steps += 1
        if self.steps >= self.maxSteps: # Truncate
            return observation, reward, terminated, True, info

        return observation, reward, terminated, False, info
    
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
        if self.cell_q_values is not None:
            color_low = np.array([255, 255, 255])  # White for low Q-values
            color_high = np.array([255, 100, 100])  # Red for high Q-values
            for i in range(self.size):
                for j in range(self.size):
                    # Get the normalized q_value for this cell
                    q_value = self.cell_q_values[i, j]

                    # Interpolate the color based on the q_value (between 0 and 1)
                    color = color_high * q_value + color_low * (1 - q_value)

                    # Draw the background rectangle for this cell
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),  # Position of the cell
                            (pix_square_size, pix_square_size),  # Size of the cell
                        ),
                    )

        # Display forbinned coordinates
        if self.forbiddenCoordinates is not None:
            for cell in self.forbiddenCoordinates:
                pygame.draw.rect(
                    canvas,
                    (255, 255, 255-255*self.isSupervisorPresent),
                    pygame.Rect(
                        pix_square_size * cell,  # Position of the cell
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
        
        # Display target
        pygame.draw.circle(
            canvas,
            self.targetColor,
            (self._target_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        # Display agent
        pygame.draw.circle(
            canvas,
            self.agentColor,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Display good coins
        for goodCoin in self.goodCoinCoordinates:
            pygame.draw.circle(
                canvas,
                self.goodCoinColor,
                (goodCoin + 0.5) * pix_square_size,
                pix_square_size / 4,
            )
        # Display bad coins
        for badCoin in self.badCoinCoordinates:
            pygame.draw.circle(
                canvas,
                self.badCoinColor,
                (badCoin + 0.5) * pix_square_size,
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
    
    # Returns the average Q-value of all size*size cell given a specific goal position
    def setCellQValues(self, q_values, goalPosition, normalize=True):
        if q_values is None:
            return

        # Initialize the Q-values and counters for averaging
        self.cell_q_values = np.zeros((self.size, self.size))
        count = np.zeros((self.size, self.size))

        # Filter q_values to only include entries with the desired goal position
        filtered_q_values = {k: v for k, v in q_values.items() if np.array_equal(k[1], goalPosition)}

        for (cell_coords, goal_coords), q_values in filtered_q_values.items():
            for action, direction in self._action_to_direction.items():
                target_coords = np.array(cell_coords) + direction

                # Ensure the target coordinates are within bounds
                if (0 <= target_coords[0] < self.size) and (0 <= target_coords[1] < self.size):
                    self.cell_q_values[target_coords[0], target_coords[1]] += q_values[action]
                    count[target_coords[0], target_coords[1]] += 1

        # Avoid division by zero, keep cells with no updates at 0
        nonzero_counts = count > 0
        self.cell_q_values[nonzero_counts] /= count[nonzero_counts]

        if normalize:
            self.cell_q_values = (self.cell_q_values - np.min(self.cell_q_values)) / (np.max(self.cell_q_values) - np.min(self.cell_q_values))
        # Transpose to correct format
        self.cell_q_values = np.transpose(self.cell_q_values)

        # Do the square root of the Q-values to make the visualization more clear
        self.cell_q_values = np.sqrt(self.cell_q_values)
    
        # Print nicely, rounded to 5 decimal places
        for row in self.cell_q_values:
            print([round(cell, 5) for cell in row])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
