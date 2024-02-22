import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch

class Node:
    def __init__(self, label):
        self.label = label
        self.coordinates : tuple
        self.neighbours = [] # adjacency_list
        self.weights = {} # store weight values in dictionary {node_labe: weight}

    def add_neighbour(self, new_neighbour):
        self.neighbours.append(new_neighbour)



class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def add_node(self, new_node:Node):
        self.nodes.add(new_node)

    def plot_graph(self, custom_xlim = (0,1), custom_ylim = (0,1), axis=False, color = 'green', node_tag = True):

        # check if edges have been generated
        if self.edges:
            self.generate_edges()

        # get graph size
        N = len(self.nodes)

        # initialize plt figure
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()

        # make slightly bigger x_lim and y_lim for better visualization
        x_lim = custom_xlim[0]- (custom_xlim[1]/20) , custom_xlim[1]+ (custom_xlim[1]/20)
        y_lim = custom_ylim[0]- (custom_ylim[1]/20) , custom_ylim[1]+ (custom_ylim[1]/20)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # calculate appropriate node radius based on graph size (N) and bounding box (x_lim,y_lim)
        node_radius = (custom_xlim[1] - custom_xlim[0]) / (5 * np.sqrt(N))
        edge_lw = min((custom_xlim[1] - custom_xlim[0]) / (2 * np.sqrt(N)), 0.2)

        # draw nodes
        for node in self.nodes:
            ax.add_patch(Circle(xy=node.coordinates, radius= node_radius, color = color, alpha=.5))
            if node_tag:
                plt.text(*node.coordinates, str(node.label), size=7, ha='center', va='center',alpha=.7)
        # draw edges
        for edge in self.edges:
            print(edge[0],"------",edge[1])
            ax.add_patch(ConnectionPatch(edge[0],edge[1],'data',lw=edge_lw,color='grey'))

        plt.axis(axis)
        plt.show();

    def generate_edges(self):
        self.edges.clear() # reset
        for node in self.nodes:
            for neighbour_label in node.neighbours:
                neighbour_node = next((n for n in self.nodes if n.label == neighbour_label), None)
            if neighbour_node is not None:
                edge = (node.coordinates, neighbour_node.coordinates)
                self.edges.add(edge)

    def generate_circular_coordinates(self, center=(.5,.5), radius=.5):
        N = len(self.nodes)
        cx, cy = center
        for i, node in enumerate(self.nodes):
            angle = 2* np.pi * i / N
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            node.coordinates = (x,y)

    def generate_random_coordinates(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
        for node in self.nodes:
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            node.coordinates = (x,y)
