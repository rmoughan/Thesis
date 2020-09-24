import numpy as np
from collections import defaultdict

import pdb

class SubgoalBuffer: #need coooler name

    def __init__(self, size = 1e6):
        self.size = size
        self.stupid = set()
        self.buffer = defaultdict(lambda: [])
        self.goals = []
        self.minimum_goal_dist = 10

    def store(self, loc, action, reward):
        self.stupid.add(tuple([loc, action, reward]))

        if len(self.stupid) > self.size:
            print("should fix the size of the subgoal buffer")

    def find_subgoals(self):
        for memory in self.stupid:
            self.buffer[memory[0]].append(memory[1:])

        for key, values in self.buffer.items():
            unique_arps = set([value[1] for value in values])
            if (len(unique_arps) > 1) and not self.near_any_subgoals(key):
                if key[0] <= 65 and key[1] <= 115:
                    pdb.set_trace()
                self.goals.append(key)
        return self.goals

    def near_any_subgoals(self, loc):
        for goal in self.goals:
            dist = np.linalg.norm(np.array(goal) - np.array(loc))
            if dist <= self.minimum_goal_dist:
                return True
        return False
