import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../envs/')
from montezuma_wrapper import MontezumaWrapper
sys.path.append('../infrastructure/')
from spatial_graph import SpatialGraph
import pickle

import pdb

env = gym.make('MontezumaRevengeNoFrameskip-v4')
env = MontezumaWrapper(env)

#seed everything
random.seed(42)
np.random.seed(42)
env.seed(42)
env.action_space.seed(42)

SpatialGraph = SpatialGraph(start_node = (80, 83))
hippo = {}

hippo[(80,83)] = [0]
hippo[(65, 121)] = [5 for _ in range(34)] + [4 for _ in range(9)]
hippo[(86, 121)] = [5 for _ in range(34)] + [3 for _ in range(18)]
hippo[(101, 85)] = [3 for _ in range(5)] + [6] + [0 for _ in range(40)]

hippo[(67, 121)] = [5 for _ in range(34)] + [4 for _ in range(9)]
hippo[(85, 121)] = [5 for _ in range(34)] + [3 for _ in range(18)]
hippo[(101, 85)] = [3 for _ in range(5)] + [6] + [0 for _ in range(40)]

hippo[(76, 121)] = [5 for _ in range(34)] + [4 for _ in range(2)]
hippo[(112, 112)] = [3 for _ in range(2)] + [6] + [0 for _ in range(40)]
hippo[(66, 84)] = [4 for _ in range(12)]
hippo[(87, 121)] = [5 for _ in range(34)] + [3 for _ in range(20)]
hippo[(60, 121)] = [5 for _ in range(34)] + [4 for _ in range(12)]
hippo[(44, 85)] = [4 for _ in range(12)] + [7] + [0 for _ in range(27)] + [4 for _ in range(5)]

hippo[(65, 124)] = [5 for _ in range(34)] + [4 for _ in range(10)]
hippo[(75, 121)] = [5 for _ in range(34)] + [4 for _ in range(2)]
hippo[(112, 106)] = [3 for _ in range(2)] + [6] + [0 for _ in range(40)]
hippo[(123, 125)] = [3 for _ in range(2)] + [6] + [0 for _ in range(40)] + [6] + [0 for _ in range(30)] + [4 for _ in range(8)]
hippo[(114, 167)] = [3 for _ in range(2)] + [6] + [0 for _ in range(40)] + [6] + [0 for _ in range(30)] + [5 for _ in range(40)] + [4 for _ in range(21)]
hippo[(98, 161)] = [3 for _ in range(2)] + [6] + [0 for _ in range(40)] + [6] + [0 for _ in range(30)] + [5 for _ in range(40)] + [4 for _ in range(39)]
hippo[(72, 161)] = [3 for _ in range(2)] + [6] + [0 for _ in range(40)] + [6] + [0 for _ in range(30)] + [5 for _ in range(40)] + [4 for _ in range(39)] + \
                    [7] + [0 for _ in range(30)] + [7] + [0 for _ in range(20)] #this one is a bit off but I wanted to get it past the skull
hippo[(59, 160)] = [3 for _ in range(2)] + [6] + [0 for _ in range(40)] + [6] + [0 for _ in range(30)] + [5 for _ in range(40)] + [4 for _ in range(39)] + \
                    [7] + [0 for _ in range(30)] + [7] + [0 for _ in range(20)]

# obs = env.reset()
# test = (59, 160)
# actions = hippo[test]
# for i in range(len(actions)):
#     obs, reward, done, info = env.step(actions[i])
#
# print(info["loc"])
# from PIL import Image
# Image.fromarray(obs.squeeze()).show()
# Image.fromarray(env.get_frame()).show()
# pdb.set_trace()

unexplored_nodes = []
subgoals = []
start_node = (80, 83) #just the starting location of ALE
unexplored_nodes.append(start_node)
subgoals.append(start_node)

num_random_episodes = 200
num_actions_per_episode = 10000000000 #arbitrarily large for now


if True:
    infile = open("SpatialGraph_200_inf_iter9", "rb")
    SpatialGraph = pickle.load(infile)
    SpatialGraph.minimum_goal_dist = 20
    subgoals = SpatialGraph.find_subgoals()
    #exploration order I'm using: (80, 83), (76, 121), (6 0, 121), (87, 121), (66, 84), (44, 85)
    #new exploration order: (80, 83), (65, 124), (75, 121), (86, 121), (112, 106), (123, 125), (114, 167), (98, 161), (72, 161)
    unexplored_nodes = [(59, 160)] #just need to change this

while len(unexplored_nodes) != 0:
    current_node = unexplored_nodes.pop(0)
    print(current_node)
    total_reward = 0
    for i in range(num_random_episodes):
        print(i)
        print(len(subgoals))
        obs = env.reset()
        actions_to_current_node = hippo[current_node]
        for j in range(len(actions_to_current_node)):
            obs, reward, done, info = env.step(actions_to_current_node[j])

        total_reward = 0
        for t in range(num_actions_per_episode):
            action = np.random.randint(0,8)
            obs, reward, done, info = env.step(action)
            jumping, inAir, lives, loc = info["jumping"], info["inAir"], info["ale.lives"], info["loc"]
            total_reward += reward
            at_any_goal = SpatialGraph.near_any_other_subgoal(loc, current_node)
            done = done or lives == 5 or at_any_goal or total_reward == 400
            if jumping:
                jump_outcome = info["jump_outcome"]
                SpatialGraph.store(loc, action, jump_outcome)
                if jump_outcome == -1:
                    break
            if (not jumping) and (not inAir):
                reward = reward if lives == 6 else -1
                SpatialGraph.store(loc, action, reward)
            if at_any_goal:
                goal = at_any_goal
                if not SpatialGraph.has_edge(current_node, goal):
                    SpatialGraph.add_edge(current_node, goal)
            if done:
                break
        new_subgoals = SpatialGraph.find_subgoals()
        set_diff = [i for i in new_subgoals if i not in subgoals]
        for ns in set_diff:
            SpatialGraph.add_node(ns)
            SpatialGraph.add_edge(current_node, ns)
            unexplored_nodes.append(ns)
        subgoals = new_subgoals.copy()

    plt.subplot(121)
    env.reset()
    obs = env.get_frame()
    plt.imshow(obs)

    for subgoal_location in new_subgoals: #should probably just make this a method at this point
        plt.scatter(subgoal_location[0], subgoal_location[1], c = "yellow", alpha = 0.5, s = 10)
    plt.show()

    SpatialGraph.draw_pretty()
    SpatialGraph.clean_graph()
    SpatialGraph.draw_pretty()

    if True:
        outfile = open("SpatialGraph_200_inf_iter10_min20", "wb")
        pickle.dump(SpatialGraph, outfile)
        outfile.close()

    pdb.set_trace()



#would be good to come back to all of the subgoals and do random walks for a bit just to check to make sure there
#aren't any edges missing. Kinda want to do this at the end of the 7th iteration just so I can feel a bit sane.
# unexplored_nodes = SpatialGraph.nodes
# unexplored_nodes = [(80, 83), (76, 121), (60, 121), (87, 121), (66, 84), (44, 85), (112, 112)]
# unexplored_nodes = [(112, 112)]
# while len(unexplored_nodes) != 0:
#     current_node = unexplored_nodes.pop(0)
#     print(current_node)
#     total_reward = 0
#     for i in range(1000):
#         print(i)
#         obs = env.reset()
#         actions_to_current_node = hippo[current_node]
#         for j in range(len(actions_to_current_node)):
#             obs, reward, done, info = env.step(actions_to_current_node[j])
#         for t in range(num_actions_per_episode):
#             action = np.random.randint(0,8)
#             obs, reward, done, info = env.step(action)
#             jumping, inAir, lives, loc = info["jumping"], info["inAir"], info["ale.lives"], info["loc"]
#             total_reward += reward
#             at_any_goal = SpatialGraph.near_any_other_subgoal(loc, current_node)
#             done = done or lives == 5 or at_any_goal
#             if jumping and info["jump_outcome"] == -1:
#                 break
#             if at_any_goal:
#                 goal = at_any_goal
#                 if not SpatialGraph.has_edge(current_node, goal):
#                     print("adding edge: ", current_node, "<--------->", goal)
#                     SpatialGraph.add_edge(current_node, goal)
#             if done:
#                 break


#     SpatialGraph.draw_pretty()
#     pdb.set_trace()
#     SpatialGraph.clean_graph()
#     SpatialGraph.draw_pretty()
#
# SpatialGraph.draw_pretty()
# SpatialGraph.add_edge((112, 112), (76, 121))
# SpatialGraph.add_edge((112, 112), (87, 121))
# SpatialGraph.add_edge((112, 112), (122, 128))
# SpatialGraph.add_edge((122, 128), (112, 112))
# SpatialGraph.add_edge((122, 128), (87, 121))
# SpatialGraph.add_edge((80, 83), (44, 85))
# SpatialGraph.draw_pretty()
# SpatialGraph.clean_graph()
# SpatialGraph.draw_pretty()



#what if I assumed every edge was bidirectional, then tested each node like above and modified edges to be unidirectional where you
#never reach the node? That could work... maybe?

#TODO:
#keep running and checking iterations. Just need that last piece...

#Future interesting ideas:
# allow subgoals to be relative to other obejcts, not just fixed coordinates. E.g. "I am two feet from the skull". This could
# make it so that problems look similar, which would allow for some serious generalizability
