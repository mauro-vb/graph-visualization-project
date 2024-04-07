from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import matplotlib.pyplot as plt

COLOUR = "#98c412"

class Node:
    """
    Represents a node in a graph, encapsulating properties like ID, number,
    and connections to other nodes. It also includes a visual representation
    using matplotlib for plotting purposes.
    """
    def __init__(self, _id, number=None, colour=COLOUR):
        """
        Initialises a new instance of the Node class.

        Parameters:
        - _id: The unique identifier for the node.
        - number: An optional numerical value associated with the node. (Useful to compute Distance Matrix)
        - colour: The display colour of the node in the plot, defaulting to 'green'.
        """
        self.id = _id
        self.number = number
        self.in_neighbours = []  # Nodes from which this node can be reached.
        self.out_neighbours = []  # Nodes that can be reached from this node.
        # Circle representation for plotting, with a semi-transparent overlay.
        self.circle = Circle(xy=np.zeros(2), radius=0.1, color=colour, alpha=.9, zorder=100)


    def add_out_neighbour(self, neighbour, weight=1):
        if neighbour not in self.out_neighbours:
            self.out_neighbours.append((neighbour, weight))

    def add_in_neighbour(self, neighbour, weight=1):
        if neighbour not in self.in_neighbours:
            self.in_neighbours.append((neighbour, weight))

    def has_neighbour(self,neighbour):
        if neighbour in [node for node,_ in self.out_neighbours]:
            return True
        if neighbour in [node for node,_ in self.in_neighbours]:
            return True
        else:
            return False

    def degree(self):
        return len(self.out_neighbours) + len(self.in_neighbours)

    def in_degree(self):
        return len(self.in_neighbours)
    def out_degree(self):
        return len(self.out_neighbours)

    def show_label(self, ax):
        ax.text(*self.circle.center, str(self.id), size=min(10,450*self.circle.radius), ha='center', alpha=.8,color="black")

class TreeNode(Node):
    """
    Represents a node within a tree structure, extending the basic Node class
    with tree-specific properties such as parent-child relationships and
    spatial attributes like width, level, and height.
    NOTE: TreeNode can represent a full Tree by being its root.
    """
    def __init__(self, label, tree_dict=None):
        """
        Initialises a new instance of the TreeNode class.

        Parameters:
        - label: The unique label or identifier for the tree node.
        - tree_dict: An optional dictionary representing a tree structure from which
          this node and its children can be initialised.
        """
        super().__init__(label)
        self.children = set()  # Children of this node, stored in a set to avoid duplicates.
        self.parent = None  # The parent node of this node. If None this is a root
        self.width = None  # Width of the subtree rooted at this node, for layout purposes.
        self.level = None  # The level of this node in the tree.
        self.height = None  # Height of the subtree rooted at this node.
        self.x_y_ratio = None  # Aspect ratio for visual representation purposes.

    @classmethod
    def build_tree_from_dict(cls, tree_dict):
        """
        Constructs a tree from a dictionary representation, linking nodes based on parent-child relationships.

        This class method facilitates the creation of a tree structure from a dictionary, where each key represents a node
        label, and its values are the labels of its children. It uses a recursive helper function to ensure that all nodes
        are created and properly linked from the root downwards.
        """
        # Create a dictionary to store TreeNode instances by label
        node_instances = {}

        # Recursively build the tree
        def build_tree_helper(label):
            # Create TreeNode if label doesn't exist as one yet
            if label not in node_instances:
                node_instances[label] = TreeNode(label)

            node = node_instances[label]

            # Add children to the TreeNode if it has any
            if label in tree_dict:
                for child_label in tree_dict[label]:
                    child_node = build_tree_helper(child_label) # Recursively add children's children to the dictionary
                    node.add_child(child_node)

            return node # Return TreeNode

        # Start building the tree from the root node
        root_label = next(iter(tree_dict)) # First key in the dictionary
        root_node = build_tree_helper(root_label)

        return root_node

    def add_child(self, child, weight=1):
        """
        Adds a child node to this node's set of children.

        Sets up a parent-child relationship between this node and the child.
        This method is crucial for building the tree structure by connecting nodes.
        """
        child.parent = self
        self.children.add((child, weight))

    def get_height(self):
        """
        Calculates the height of the tree rooted at this node.

        The height is defined as the number of edges on the longest downward path between this node and a leaf node.
        """
        # Base case, if the node has no chidlren teh height is 0.
        if not self.children:
            return 0
        # Recursively find the height of each child and add 1 to account for the current node's level.
        self.height = 1 + max([child[0].get_height() for child in self.children])
        return self.height

    def compute_drawing_params(self):
        """
        Calculates necessary parameters for drawing the tree: width, height, and the aspect ratio for visual representation.
        """
        self.calculate_width()
        self.get_height()
        self.x_y_ratio = max((self.width//2) // self.height, self.height // (self.width//2))

    def get_level(self):
        """
        Finds the level of the current node within the tree.

        The level is the distance from the root node, with the root being at level 0.
        """
        self.level = 0 # Start at level 0
        current_parent = self.parent # 1st parent
        while current_parent:
            current_parent = current_parent.parent  # Keep looking for parent of parent
            self.level += 1 # Increase the level each time
        return self.level

    def calculate_width(self):
        """
        Calculates the width of the subtree rooted at this node, defined as
        the sum of the widths of its children.
        """
        self.width = 0
        # Recruseively calculate width of children
        children_widths = [child[0].calculate_width() for child in self.children]
        self.width = 1 + sum(children_widths) # 1 + to avoid division by 0
        return self.width

    def compute_coordinates(self, x, y, x_y_ratio=2):
        """
        Computes and assigns coordinates to each node for visual representation, starting with the root node.
        """
        self.coordinates = (x, y) # Set root node coordinates
        if self.children:
            total_children_width = sum([child[0].width for child in self.children]) # Calculate total width
            starting_x = x - total_children_width / 2  # Starting x for the children, current node's x minus half of the total width
            # Position each child and recursively calculate their coordinates.
            for child, _ in self.children:
                child_x = starting_x + child.width / 2 # Get x for each child based on starting x and half their own width (as to be centered)
                child.compute_coordinates(child_x, 0 - child.get_level() * x_y_ratio, x_y_ratio) # Adjust y (layers of teh tree)
                starting_x += child.width # Update starting x for next child

    def draw_tree(self, labels=True, non_tree_edges=None, show_layers=True,colour=COLOUR):
        """
        Draws the tree using matplotlib, including optional labels for each node and additional non-tree edges.
        """
        self.compute_drawing_params()
        self.compute_coordinates(0, 0, self.x_y_ratio)
        fig = plt.figure()
        ax = fig.gca()
        plt.axis(False) # Hide the axis for a cleaner display.

        drawn_edges=[]

        def draw_patch(node, ax, zorder=1):
            # Draw node.
            ax.add_patch(Circle(xy=node.coordinates, radius=0.5, color=colour, alpha=1, zorder=zorder))
            if labels:
                plt.text(*node.coordinates, str(node.id), size=6, ha='center', va='baseline', alpha=.7,zorder=zorder+1)
            # Draw edges if it has children.
            if node.children:
                for child, _ in node.children:
                    ax.add_patch(ConnectionPatch(node.coordinates, child.coordinates, 'data', lw=0.3, color='grey',zorder=-zorder))
                    drawn_edges.append((node.id, child.id))
                    draw_patch(child, ax,zorder+1) # Recursively draw children and their edges.

            # Draw non-tree Edges if provided.
            if non_tree_edges:
                for nt_neighbour_label in non_tree_edges[node.id]:
                    nt_neighbour_node = self.find_tree_node(nt_neighbour_label)
                    if nt_neighbour_node and (node.id, nt_neighbour_label) not in drawn_edges and (nt_neighbour_label,node.id) not in drawn_edges:
                        ax.add_patch(ConnectionPatch(node.coordinates, nt_neighbour_node.coordinates, 'data', lw=0.1, color='blue', linestyle=":"))

        draw_patch(self, ax) # Start drawing from the root.

        # Draw horizontal lines at each layer
        if show_layers:
            for layer_y in [- i * self.x_y_ratio for i in range(self.height + 1)]:
                ax.add_patch(ConnectionPatch((- self.width/2, layer_y), (self.width/2, layer_y), 'data', lw=0.2, color='grey'))

        # Set Figure boundaries
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
        """
        Searches for and returns a node with the given label, within this tree.
        """
        queue = [self]
        while queue:
            current_node = queue.pop(0)
            if current_node.id == label:
                return current_node
            queue.extend([child[0] for child in current_node.children])
