import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import pdb

class SpatialGraph:

    def __init__(self, size = 1e6, start_node = None):
        self.size = size
        # self.buffer = defaultdict(lambda: [])
        self.buffer = {}
        self.nodes = []
        self.graph = nx.DiGraph()
        if start_node != None:
            self.nodes.append(start_node)
            self.graph.add_node(start_node)
        self.minimum_goal_dist = 20

    def store(self, loc, action, reward):

        if loc not in self.buffer.keys(): #had to get rid of the defaultdict for pickle to work
            self.buffer[loc] = []

        if reward not in self.buffer[loc] and loc != (-1, -1):
            self.buffer[loc].append(reward)

        if len(self.buffer) > self.size:
            raise ValueError("SpatialGraph Buffer Overflow. Consider changing the size")

    def find_subgoals(self):
        for key, values in self.buffer.items():
            if (len(values) > 1) and not self.near_any_subgoals(key):
                self.nodes.append(key)
        return self.nodes

    def near_any_subgoals(self, loc):
        for goal in self.nodes:
            dist = np.linalg.norm(np.array(goal) - np.array(loc))
            if dist <= self.minimum_goal_dist:
                return True
        return False

    def near_any_other_subgoal(self, loc, current_node):
        for goal in self.nodes:
            if goal == current_node:
                continue
            dist = np.linalg.norm(np.array(goal) - np.array(loc))
            if dist <= 5:
                return goal
        return None

    def add_edge(self, src, dest):
        if src == (80, 83) and dest == (112, 106): #a bit hacky for now, but there are more interesting things to spend my time on
            self.graph.add_edge(src, dest)
        else:
            self.graph.add_edge(src, dest)
            self.graph.add_edge(dest, src)

    def has_edge(self, src, dest):
        return self.graph.has_edge(src, dest)

    def add_node(self, node):
        self.graph.add_node(node)

    def clean_graph(self):
        for node in self.graph.nodes:
            self._clean_node(node)

    def _clean_node(self, starting_node):
        #General idea: Look at all bidirectional edges from a starting node.
        #Find the path length to all the other nodes using these edges, and if there
        #exists paths of length 1 and more than 1, the path of length 1 is one that in
        #theory wouldn't be learned in an ideal setting. Check to see if it is the longest
        #leg in that cycle, and if so, remove.

        #Possibly more general idea than just triangles. If there ever exists a single step
        #path and a non-single step path to a node via bidrectional edges and the single step
        #path is a the longest edge, cut that edge

        all_paths = defaultdict(lambda: [])
        for edge in self._bidirectional_edges(starting_node):
            neighbor_node = edge[-1]
            paths = self.paths_from_edge(starting_node, neighbor_node, {starting_node: []})
            for n,path in paths.items():
                if path not in all_paths[n]:
                    all_paths[n].append(path)

        for no, paths in all_paths.items():
            if len(paths) > 1:
                path_lengths = [len(p) for p in paths]
                if 1 in path_lengths and max(path_lengths) > 1:
                    edge_lengths = [self._edge_lengths(starting_node, p) for p in paths]
                    edge_lengths = sorted(edge_lengths, key = lambda x: len(x))
                    candidate = edge_lengths[0][0]
                    rest = edge_lengths[1:]
                    max_rest = max([max(d) for d in rest])
                    if candidate > max_rest:
                        print("cutting edge: ", starting_node, "<--------->", no)
                        self.graph.remove_edge(starting_node, no)
                        self.graph.remove_edge(no, starting_node)




    def paths_from_edge(self, prev_node, node, table):
        table[node] = table[prev_node] + [node]
        edges = self._bidirectional_edges(node)
        neighbor_nodes = [e[-1] for e in edges]
        nodes_left = [n for n in neighbor_nodes if n not in table.keys()]

        if len(nodes_left) == 0:
            return table

        for n in nodes_left:
            self.paths_from_edge(node, n, table)
        return table

    def _bidirectional_edges(self, node):
        bidirectional_edges = []
        for edge in self.graph.edges(node):
            if edge[::-1] in self.graph.edges:
                bidirectional_edges.append(edge)
        return bidirectional_edges

    def _edge_lengths(self, start_node, path):
        lengths = []
        for p in path:
            d = np.linalg.norm(start_node - np.array(p), ord = 2)
            lengths.append(d)
            start_node = np.array(p)
        return lengths

    def draw(self):
        plt.subplot(111)
        nx.draw_circular(self.graph, with_labels=True, font_weight='bold', font_size = 6)
        plt.show()

    def draw_pretty(self):
        plt.figure(figsize = (9,11))
        # env.reset()
        # obs = env.get_frame()
        obs = np.zeros(shape = (210,160,3))
        plt.imshow(obs)
        for node in self.nodes:
            plt.scatter(node[0], node[1], c = "yellow", alpha = 1.0, s = 50)
        for edge in self.graph.edges:
            n1, n2 = edge
            x1, y1 = n1
            x2, y2 = n2
            dx, dy = x2 - x1, y2 - y1
            if edge[::-1] in self.graph.edges:
                plt.arrow(x1, y1, dx, dy, head_width = 2.0, head_length = 1, color = "green", length_includes_head = True)
            else:
                plt.arrow(x1, y1, dx, dy, head_width = 2.0, head_length = 1, color = "red", length_includes_head = True)

        plt.show()
