import numpy as np
from collections import defaultdict

import pdb

class SubgoalBuffer:

    def __init__(self, size = 1e6):
        self.size = size
        self.buffer = defaultdict(lambda: [])
        self.goals = []
        self.minimum_goal_dist = 10

    def store(self, loc, action, reward):
        #kinda want to round loc
        if reward not in self.buffer[loc]:
            self.buffer[loc].append(reward)

        if len(self.buffer) > self.size:
            raise ValueError("SpatialGraph Buffer Overflow. Consider changing the size")

    def find_subgoals(self):
        for key, values in self.buffer.items():
            if (len(values) > 1) and not self.near_any_subgoals(key):
                self.goals.append(key)
        return self.goals

    def near_any_subgoals(self, loc):
        for goal in self.goals:
            dist = np.linalg.norm(np.array(goal) - np.array(loc))
            if dist <= self.minimum_goal_dist:
                return True
        return False
