
import gymnasium as gym
from gridWorldEnv import GridWorldEnv

env = GridWorldEnv(render_mode="human", size=5)
observation, info = env.reset()
print(f"observation: {observation}, info: {info}")
print(f"action_space: {env.action_space}, observation_space: {env.observation_space}")
action = env.action_space.sample()  # agent policy that uses the observation and info
observation, reward, terminated, truncated, info = env.step(action)
env.close()
print(f"action: {action}")
print(f"observation: {observation}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
"""
observation: {'agent': array([4, 3]), 'target': array([2, 2])}, info: {'distance': 3.0}
action_space: Discrete(4), observation_space: Dict('agent': Box(0, 4, (2,), int32), 'target': Box(0, 4, (2,), int32))
action: 1
observation: {'agent': array([4, 4]), 'target': array([2, 2])}, reward: 0, terminated: False, truncated: False, info: {'distance': 4.0}
"""

"""
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
"""