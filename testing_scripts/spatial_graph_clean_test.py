import sys
sys.path.append('../infrastructure/')
from spatial_graph import SpatialGraph
import networkx as nx

import pdb

SpatialGraph = SpatialGraph()
G = nx.DiGraph()

A = (80, 83)
B = (76, 121)
C = (112, 112)
D = (87, 121)
E = (66, 84)
F = (60, 121)

G.add_node(A)
G.add_node(B)
G.add_node(C)
G.add_node(D)
G.add_node(E)
G.add_node(F)

G.add_edge(A, E)
G.add_edge(E, A)

G.add_edge(A, B)
G.add_edge(B, A)

G.add_edge(A, C)

G.add_edge(F, B)
G.add_edge(B, F)

G.add_edge(D, B)
G.add_edge(B, D)

G.add_edge(D, C)
G.add_edge(C, D)

G.add_edge(B, C)
G.add_edge(C, B)

SpatialGraph.graph = G

SpatialGraph.draw()
SpatialGraph.clean_graph()
SpatialGraph.draw()
