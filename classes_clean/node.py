from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, _id, number, color='green'):
        self.id = _id
        self.number = number
        self.in_neighbours = []
        self.out_neighbours = []
        self.circle = Circle(xy=np.zeros(2), radius = .1, color = color, alpha=.5)

    def add_out_neighbour(self, neighbour, weight=1):
        if neighbour not in self.out_neighbours:
            self.out_neighbours.append((neighbour, weight))

    def add_in_neighbour(self, neighbour, weight=1):
        if neighbour not in self.in_neighbours:
            self.in_neighbours.append((neighbour, weight))

    def show_label(self, ax):
        ax.text(*self.circle.center, str(self.id), size=6, ha='center', va='baseline', alpha=.5)

class TreeNode(Node):
    def __init__(self, label, tree_dict=None):
        super().__init__(label)
        self.children = set()
        self.parent = None
        self.non_tree_neighbours = []
        self.width = None
        self.level = None
        self.height = None
        self.x_y_ratio = None

    @classmethod
    def build_tree_from_dict(cls, tree_dict):
        node_instances = {}

        def build_tree_helper(label):
            if label not in node_instances:
                node_instances[label] = TreeNode(label)

            node = node_instances[label]

            if label in tree_dict:
                for child, weight in tree_dict[label]:
                    child_node = build_tree_helper(child.id)
                    node.add_child(child_node, weight)

            return node

        root_label = next(iter(tree_dict))
        root_node = build_tree_helper(root_label)

        return root_node

    def add_child(self, child, weight=1):
        child.parent = self
        self.children.add((child, weight))

    def get_height(self):
        if not self.children:
            return 0
        self.height = 1 + max([child[0].get_height() for child in self.children])
        return self.height

    def compute_drawing_params(self):
        self.calculate_width()
        self.get_height()
        self.x_y_ratio = max((self.width//2) // self.height, self.height // (self.width//2))

    def get_level(self):
        self.level = 0
        current_parent = self.parent
        while current_parent:
            current_parent = current_parent.parent
            self.level += 1
        return self.level

    def calculate_width(self):
        self.width = 0
        children_widths = [child[0].calculate_width() for child in self.children]
        self.width = 1 + sum(children_widths)
        return self.width

    def compute_coordinates(self, x, y, x_y_ratio=1):
        self.coordinates = (x, y)
        if self.children:
            total_children_width = sum([child[0].width for child in self.children])
            starting_x = x - total_children_width // 2
            for child, _ in self.children:
                child_x = starting_x + child.width // 2
                child.compute_coordinates(child_x, 0 - child.get_level() * x_y_ratio, x_y_ratio)
                starting_x += child.width

    def draw_tree(self, labels=True, non_tree_edges=False):
        self.compute_drawing_params()
        self.compute_coordinates(0, 0, self.x_y_ratio)
        fig = plt.figure()
        ax = fig.gca()
        plt.axis(False)

        def draw_patch(node, ax):
            ax.add_patch(Circle(xy=node.coordinates, radius=0.5, color='green', alpha=0.3))
            if labels:
                plt.text(*node.coordinates, str(node.id), size=6, ha='center', va='baseline', alpha=0.5)
            if node.children:
                for child, _ in node.children:
                    ax.add_patch(ConnectionPatch(node.coordinates, child.coordinates, 'data', lw=0.5, color='grey'))
                    draw_patch(child, ax)
            if non_tree_edges:
                for nt_neighbour_label in node.non_tree_neighbours:
                    nt_neighbour_node = self.find_tree_node(nt_neighbour_label)
                    ax.add_patch(ConnectionPatch(node.coordinates, nt_neighbour_node.coordinates, 'data', lw=0.1, color='blue', linestyle=":"))

        draw_patch(self, ax)

        margin = 2
        x_lim = self.width // 2
        ax.set_xlim((-x_lim - margin, x_lim + margin))
        ax.set_ylim((-self.height * self.x_y_ratio - margin, 0 + margin))
        plt.grid(which='minor', axis='y', linewidth=0.5, linestyle=':')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        ax.tick_params(tick1On=False, which='both')
        ax.minorticks_on()
        plt.show()
        return fig

    def find_tree_node(self, label):
        queue = [self]
        while queue:
            current_node = queue.pop(0)
            if current_node.id == label:
                return current_node
            queue.extend([child[0] for child in current_node.children])
