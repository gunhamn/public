
import numpy as np
import pygame
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, randomSpawn=True, q_values=None, goalReward=50, stepLoss=-1, wallCells=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.randomSpawn = randomSpawn
        self.q_values = q_values
        self.cell_q_values = np.zeros((self.size, self.size)) # Q-values for each cell and action
        self.goalReward = goalReward
        self.stepLoss = stepLoss

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
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

    def _get_obs(self): # Wall
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if self.randomSpawn: #Wall
            # Choose the agent's location uniformly at random
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

            # We will sample the target's location randomly until it does not coincide with the agent's location
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )
        else:
            self._agent_location = np.array([0, 0])
            self._target_location = np.array([self.size - 1, self.size - 1])

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = self.goalReward if terminated else self.stepLoss
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

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

        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._target_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 4,
        )

        # Finally, add some gridlines
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
    
