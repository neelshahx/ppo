import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-DoorKey-16x16-v0", render_mode="human")
observation, info = env.reset()
episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
env.close()
