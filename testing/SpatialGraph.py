import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../envs/')
from montezuma_wrapper import MontezumaWrapper
sys.path.append('../infrastructure/')
from subgoal_buffer import SubgoalBuffer
import networkx as nx

import pdb

env = gym.make('MontezumaRevengeNoFrameskip-v4')
env = MontezumaWrapper(env)

#seed everything
random.seed(42)
np.random.seed(42)
env.seed(42)

subgoal_buffer = SubgoalBuffer()

unexplored_nodes = []
all_nodes = []
start_node = blah #just the starting location of ALE
unexplored_nodes.append(start_node)
all_nodes.append(start_node)

while len(unexplored_nodes) != 0:
    current_node = unexplored_nodes.pop(0)
    obs = env.reset()
    total_reward = 0
    #navigate to current node
    for _ in range(num_random_actions):
        #explore code. Will tweak a bit, this is just a rough idea. Definitely need to modify single for loop structure
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        jumping, inAir, lives, loc = info["jumping"], info["inAir"], info["ale.lives"], info["loc"]
        total_reward += reward
        at_any_goal = any([env.at_goal(loc, goal) for goal in all_nodes])
        done = done or lives == 5 or at_any_goal or total_reward == 400
        if jumping:
            jump_outcome = info["jump_outcome"]
            subgoal_buffer.store(loc, action, jump_outcome)
        if (not jumping) and (not inAir):
            reward = reward if lives == 6 else -1
            subgoal_buffer.store(loc, action, reward)
        if done:
            break
    subgoals = SubgoalBuffer.find_subgoals()
    new_subgoals = [s for s in subgoals if s not in all_nodes] #this has issues. I'm kinda tempted to just change this entire structure.
    for ns in new_subgoals:
        #add node to graph
