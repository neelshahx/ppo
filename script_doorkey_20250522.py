# notes
#  'mission': 'use the key to open the door and then get to the goal'}
# only need to use actions 0-2 (l,r,f) until we are at key space
# only need pickup 3 when at key space
# only need toggle 5 when near door space
# never use actions 4, 6 (drop, done)

import gymnasium as gym
import minigrid
from matplotlib import pyplot as plt
plt.ion()

#%%

env = gym.make("MiniGrid-DoorKey-16x16-v0", render_mode="rgb_array")
observation, info = env.reset()


#%%

plt.imshow(env.render())