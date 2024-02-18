import pygraphviz
import numpy as np

def get_adj_matrix(dot_file_path):
    G = pygraphviz.AGraph()
    G.read(dot_file_path)
    adj_matrix = []
    for node in G.nodes():
        # get adjacency values per node (*1) turns Bool to value 1 or 0
        row = [G.has_edge(node, other_node)*1 for other_node in G.nodes()]
        adj_matrix.append(row)
    return np.array(adj_matrix)

def get_adj_list(dot_file_path):
    G = pygraphviz.AGraph()
    G.read(dot_file_path)
    adj_list = {}
    for node in G.nodes():
        adj_list[node] = [other_node for other_node in G.nodes() if G.has_edge(node, other_node)]
    return adj_list
