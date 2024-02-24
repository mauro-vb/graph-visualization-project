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

class TreeNode(Node):
    def __init__(self, label):
        super().__init__(label)
        self.children = set()
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.add(child)

    def get_level(self): # count number of parents above self
        level = 0
        current_parent = self.parent
        while current_parent: # while parent exists
            current_parent = current_parent.parent # check parent of parent recursively
            level += 1 # increase level count

        return level

    def print_tree(self):
        spaces = ' ' * self.get_level() * 1
        prefix = spaces + "|__" if self.parent else ""

        print(prefix + self.label)
        if self.children:
            for child in self.children:
                child.print_tree()


# binary tree node class
class BTNode(Node):
    def __init__(self, label):
        super().__init__(label)
        self.left = None
        self.right = None

    def add_child(self, child_node):
        if not self.left:
            self.left = BTNode(child_node)
        elif not self.right:
            self.right = BTNode(child_node)
        else:
            print("Node already has 2 children")


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = set()

    def add_node(self, new_node:Node):
        self.nodes[new_node.label] = new_node

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
        for node in self.nodes.values():
            ax.add_patch(Circle(xy=node.coordinates, radius= node_radius, color = color, alpha=.5))
            if node_tag:
                plt.text(*node.coordinates, str(node.label), size=7, ha='center', va='center',alpha=.7)
        # draw edges
        for edge in self.edges:
            ax.add_patch(ConnectionPatch(edge[0],edge[1],'data',lw=edge_lw,color='grey'))

        plt.axis(axis)
        plt.show();

    def generate_edges(self):
        self.edges.clear()  # reset edges
        for label, node in self.nodes.items():
            for neighbour_label in node.neighbours:  # Assuming 'neighbours' contains labels
                neighbour_node = self.nodes[neighbour_label]

                edge = (node.coordinates, neighbour_node.coordinates)
                self.edges.add(edge)

    def generate_circular_coordinates(self, center=(.5,.5), radius=.5):
        N = len(self.nodes)
        cx, cy = center
        i = 0
        for node in self.nodes.values():
            angle = 2* np.pi * i / N
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            node.coordinates = (x,y)
            i += 1

    def generate_random_coordinates(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
        for node in self.nodes:
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            node.coordinates = (x,y)


class Tree(Graph):
    def __init__(self):
        super().__init__()
        self.root = None
        self.parent_dict = {}

    def compute_bfs_tree(self, root_label):
        visited = set()  # for storing visited nodes
        q = [self.nodes[root_label]]  # for storing nodes to be visited
        visited.add(root_label)  # mark the start node as visited
        root_tree_node = TreeNode(root_label)  # create TreeNode for the root node
        self.root = root_tree_node  # set the root node of the tree

        while q:  # while the queue Q is not empty, continue the traversal
            current_node = q.pop(0)  # dequeue current node
            current_node_label = current_node.label

            current_tree_node = self.find_tree_node(current_node_label)  # find corresponding TreeNode

            for neighbour_label in current_node.neighbours:  # loop over neighbors of current node
                if neighbour_label not in visited:  # if neighbour has not been visited
                    q.append(self.nodes[neighbour_label])  # enqueue neighbour
                    visited.add(neighbour_label)  # mark neighbour as visited

                    # create TreeNode for the neighbour and set its parent
                    neighbour_tree_node = TreeNode(neighbour_label)
                    neighbour_tree_node.parent = current_tree_node
                    current_tree_node.add_child(neighbour_tree_node)

## NEEDS FIXNG based in TreeNode data strucuture
    def compute_dfs_tree(self, root):
        visited = set()  # for storing visited nodes
        self.parent_dict = {}


        root = self.nodes[root]

        def dfs(nodes, root):
            if root not in visited:
                visited.add(root)
                for neighbour_label in root.neighbours:
                    neighbour = self.nodes[neighbour_label]
                    self.parent_dict[neighbour] = root # set 'root' as parent
                    dfs(nodes, neighbour) # recursive call to go into depth for each neighbouring node


    def find_tree_node(self, label):
        """Find the TreeNode corresponding to the given label."""
        queue = [self.root]  # start from the root node
        while queue:
            current_node = queue.pop(0)
            if current_node.label == label:
                return current_node
            queue.extend(current_node.children)  # add children nodes to the queue
        return None

        dfs(self.nodes,root)
