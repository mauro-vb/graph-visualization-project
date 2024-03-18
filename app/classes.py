import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Circle, ConnectionPatch
from collections import deque

class Node:
    def __init__(self, label):
        self.label = label
        self.coordinates = None
        self.neighbours = [] # adjacency_list
        self.weights = {} # store weight values in dictionary {node_labe: weight}

    def add_neighbour(self, new_neighbour):
        self.neighbours.append(new_neighbour)

class TreeNode(Node):
    def __init__(self, label, tree_dict=None):
        super().__init__(label)
        self.children = set()
        self.parent = None
        self.non_tree_neighbours = []

        # for drawing
        self.width = None
        self.level = None
        self.height = None
        self.x_y_ratio = None

    @classmethod
    def build_tree_from_dict(cls, tree_dict):
        # Create a dictionary to store TreeNode instances by label
        node_instances = {}

        # Recursively build the tree
        def build_tree_helper(label):
            if label not in node_instances:
                node_instances[label] = TreeNode(label)

            node = node_instances[label]

            if label in tree_dict:
                for child_label in tree_dict[label]:
                    child_node = build_tree_helper(child_label)
                    node.add_child(child_node)

            return node

        # Start building the tree from the root node
        root_label = next(iter(tree_dict))
        root_node = build_tree_helper(root_label)

        return root_node

    def add_child(self, child):
        child.parent = self
        self.children.add(child)

    def get_height(self):
        if not self.children: # base case
            return 0
        # recursively compare childrens heights
        self.height = 1 + max([child.get_height() for child in self.children])
        return self.height

    def compute_drawing_params(self):
        self.calculate_width()
        self.get_height()
        self.x_y_ratio = max((self.width//2) // self.height, self.height // (self.width//2))

    def get_level(self): # count number of parents above self
        self.level = 0
        current_parent = self.parent
        while current_parent: # while parent exists
            current_parent = current_parent.parent # check parent of parent recursively
            self.level += 1 # increase level count

        return self.level

    def calculate_width(self):
        self.width = 0
        # Calculate width recursively for each child
        children_widths = [child.calculate_width() for child in self.children]

        # Width of current node is 1 + sum of widths of children's nodes
        self.width = 1 + sum(children_widths)
        return self.width

    def compute_coordinates(self, x, y, x_y_ratio = 1):

        self.coordinates = (x, y)
        if self.children:
            total_children_width = sum([child.width for child in self.children])
            starting_x = x - total_children_width // 2
            for child in self.children:
                child_x = starting_x + child.width // 2
                child.compute_coordinates(child_x, 0 - child.get_level() * x_y_ratio, x_y_ratio)
                starting_x += child.width

    def draw_tree(self,labels=False, non_tree_edges=False):
        self.compute_drawing_params()
        self.compute_coordinates(0,0, self.x_y_ratio)
        fig = plt.figure()
        ax = fig.gca()
        plt.axis(False)
        def draw_patch(node, ax):
            ax.add_patch(Circle(xy=node.coordinates, radius= .5, color = 'green', alpha=.3))
            if labels:
                plt.text(*node.coordinates, str(node.label), size=6, ha='center', va='baseline',alpha=.5)
            if node.children:
                for child in node.children:
                    ax.add_patch(ConnectionPatch((node.coordinates), child.coordinates,'data',lw=.5,color='grey'))
                    draw_patch(child, ax)
            if non_tree_edges:

                for nt_neighbour_label in node.non_tree_neighbours:
                    print(nt_neighbour_label)
                    nt_neighbour_node = self.find_tree_node(nt_neighbour_label)
                    ax.add_patch(ConnectionPatch((node.coordinates), nt_neighbour_node.coordinates,'data',lw=.1,color='blue',linestyle=":"))


        draw_patch(self, ax)

        margin = 2
        x_lim = self.width // 2
        ax.set_xlim((-x_lim-margin,x_lim+margin))
        ax.set_ylim((-self.height * self.x_y_ratio - margin, 0+margin))
        # plt.grid(which='minor', axis='y', linewidth=0.5, linestyle=':')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_frame_on(False)
        # ax.tick_params(tick1On=False,which='both')
        # ax.minorticks_on()
        #plt.show();
        return fig

    def find_tree_node(self, label):
        """Find the TreeNode corresponding to the given label."""
        queue = [self]
        while queue:
            current_node = queue.pop(0)
            if current_node.label == label:
                return current_node
            queue.extend(current_node.children)  # add children nodes to the queue

    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""

        print(prefix + self.label)
        if self.children:
            for child in self.children:
                child.print_tree()

    def print_coordinates(self):
        print(self.coordinates)
        if self.children:
            for child in self.children:
                child.print_coordinates()

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.min_max_x = np.zeros(2)
        self.min_max_y = np.zeros(2)

    def add_node(self, new_node:Node):
        self.nodes[new_node.label] = new_node

    def plot_graph(self, axis=False, color = 'green', node_tag = True):

        # check if edges have been generated
        if not self.edges:
            self.generate_edges()

        # get graph size
        N = len(self.nodes)

        # initialize plt figure
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()

        # calculate appropriate node radius based on graph size (N) and bounding box (x_lim,y_lim)
        node_radius = (self.min_max_x[1] - self.min_max_x[0]) / (5 * math.sqrt(N))
        edge_lw = min((self.min_max_y[1] - self.min_max_y[0]) / (2 * math.sqrt(N)), 0.2)


        x_lim = self.min_max_x + np.array([-node_radius,node_radius])
        y_lim = self.min_max_y + np.array([-node_radius,node_radius])

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)



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
        return fig

    def generate_edges(self):
        self.edges = []  # reset edges
        for label, node in self.nodes.items():
            for neighbour_label in node.neighbours:  # Assuming 'neighbours' contains labels
                neighbour_node = self.nodes[neighbour_label]

                edge = (node.coordinates, neighbour_node.coordinates)
                self.edges.append(edge)

    def generate_circular_coordinates(self, center=(.5,.5), radius=.5):
        N = len(self.nodes)
        cx, cy = center
        i = 0
        for node in self.nodes.values():
            angle = 2* np.pi * i / N
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            node.coordinates = np.array([x,y])
            i += 1


    def generate_random_coordinates(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
        for node in self.nodes.values():
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            node.coordinates = np.array([x,y])


    def force_directed_graph(self, embedder_type = "Eades", K=500, epsilon=1e-4, delta=.1, c=.9,c_rep=1,c_spring=2):

        l = c * 1  # ideal edge length

        def repulsive_force(distance, diff):
            if embedder_type == "Fruchterman & Reingold":
                return (l**2 / distance) * diff
            if embedder_type == "Eades":
                return c_rep * diff / (distance**2)


        def attractive_force(distance, diff):
            if embedder_type == "Fruchterman & Reingold":
                return (distance**2 / l) * diff
            if embedder_type == "Eades":
                return c_spring * np.log(distance / l) * diff


        for iteration in range(K):
            displacement = {v: np.zeros(2) for v in self.nodes.keys()}

            # Calculate repulsive forces
            for key_u, u in self.nodes.items():
                for v in self.nodes.values():
                    diff = u.coordinates - v.coordinates
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        disp = repulsive_force(distance, diff)
                        displacement[key_u] += disp

            # Calculate attractive forces
            for key_u, u in self.nodes.items():
                for key_v, v in self.nodes.items():
                    diff = u.coordinates - v.coordinates
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        disp = attractive_force(distance, diff)
                        displacement[key_u] -= disp
                        displacement[key_v] += disp

            # Update positions
            for key_v, v in self.nodes.items():
                length = np.linalg.norm(displacement[key_v])
                if length > 0:
                    # displacement vector is normalized
                    v.coordinates += delta * displacement[key_v] / length
                    self.min_max_x = np.array([min(v.coordinates[0],self.min_max_x[0]), max(v.coordinates[0],self.min_max_x[1])])
                    self.min_max_y = np.array([min(v.coordinates[1],self.min_max_y[0]), max(v.coordinates[1],self.min_max_y[1])])

            max_displacement = max(np.linalg.norm(disp) for disp in displacement.values())

            if max_displacement < epsilon:
                break

        #eturn positions

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
            for neighbour_label in current_node.neighbours:
                current_tree_node.non_tree_neighbours.append(neighbour_label) # track non tree related neighbour

            for neighbour_label in current_node.neighbours:  # loop over neighbors of current node
                if neighbour_label not in visited:  # if neighbour has not been visited
                    q.append(self.nodes[neighbour_label])  # enqueue neighbour
                    visited.add(neighbour_label)  # mark neighbour as visited

                    # create TreeNode for the neighbour and set its parent
                    neighbour_tree_node = TreeNode(neighbour_label)
                    neighbour_tree_node.parent = current_tree_node
                    current_tree_node.add_child(neighbour_tree_node)

    def compute_dfs_tree(self, root):
        visited = set()  # for storing visited nodes

        root = self.nodes[root]
        self.root = TreeNode(root.label)

        def dfs(nodes, root):
            current_tree_node = self.find_tree_node(root.label)

            if root.label not in visited:
                visited.add(root.label)
                for neighbour_label in root.neighbours:

                    neighbour = self.nodes[neighbour_label]
                    neighbour_tree_node = TreeNode(neighbour_label)
                    current_tree_node.non_tree_neighbours.append(neighbour_label) # track non tree related neighbour

                    if neighbour_label not in visited:
                        neighbour_tree_node.parent = current_tree_node
                        current_tree_node.add_child(neighbour_tree_node)

                    dfs(nodes, neighbour) # recursive call to go into depth for each neighbouring node
        dfs(self.nodes, root)

    def find_tree_node(self, label):
        """Find the TreeNode corresponding to the given label."""
        queue = [self.root]  # start from the root node
        while queue:
            current_node = queue.pop(0)
            if current_node.label == label:
                return current_node
            queue.extend(current_node.children)  # add children nodes to the queue
