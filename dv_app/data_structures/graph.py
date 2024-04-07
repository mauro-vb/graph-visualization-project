import pygraphviz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from collections import deque
import math

from helper_functions.edge_bundling import edge_bundling, edge_bundling_precomputed
from data_structures.node import Node
from data_structures.edge import Edge


class Graph:
    def __init__(self, incoming_dot_file=None, directed = False, subgraphs = False, a_subgraph = False, selected_subgraphs=None,weight_name="weight",colour="g"):
        """
        Initialize a Graph instance from a DOT file, supporting subgraphs and directed edges.

        Args:
            incoming_dot_file (str, optional): Path to a DOT file to load the graph from.
            directed (bool, optional): Whether the graph is directed. Defaults to False.
            subgraphs (bool, optional): Whether to consider subgraphs. Defaults to False.
            a_subgraph (bool, optional): Whether this instance represents a subgraph. Affects figure creation.
            selected_subgraphs (list, optional): Specific subgraphs to load, if not all are needed.
            weight_name (str, optional): Attribute name for edge weights in the DOT file. Defaults to "weight".
            colour (str, optional): Default colour for nodes. Defaults to "g" (green).
        """
        self.directed = directed
        self.nodes, self.edges = {}, {}
        self.subgraphs = None
        self.fig = None if a_subgraph or subgraphs else plt.figure(figsize=(7, 7))
        self.ax = None if a_subgraph or subgraphs else self.fig.gca()
        self.axes = []
        self.min_max_x, self.min_max_y = np.array([0, 1]), np.array([0, 1])
        self.dfs_order, self.bfs_order = [], []
        self.dfs_tree, self.bfs_tree = {}, {}
        self.bfs_non_tree_edges, self.dfs_non_tree_edges = None, None

        if incoming_dot_file:
            self.load_graph(incoming_dot_file, subgraphs=subgraphs, selected_subgraphs=selected_subgraphs, weight_name=weight_name, colour=colour)

    # General Purposes
    def load_graph(self, dot_file_path, subgraphs=False, selected_subgraphs=None,weight_name="weight",colour="g"):
        G = pygraphviz.AGraph()
        G.read(dot_file_path)

        if subgraphs:
            self.subgraphs = {}
            for subgraph in G.subgraphs():
                if selected_subgraphs:
                    subgraph_name = subgraph.name
                    if subgraph_name in selected_subgraphs:
                        self.subgraphs[subgraph_name] = Graph(a_subgraph=True)

                        # nodes
                        for i, graphviz_node in enumerate(subgraph.nodes()):
                            node_id = graphviz_node.get_name()
                            node = Node(_id=node_id,number=i,colour=colour)
                            self.subgraphs[subgraph_name].nodes[node_id] = node

                else:
                    subgraph_name = subgraph.name
                    self.subgraphs[subgraph_name] = Graph(a_subgraph=True)
                    # nodes
                    for graphviz_node in subgraph.nodes():
                        node_id = graphviz_node.get_name()
                        node = Node(_id=node_id,colour=colour)
                        self.subgraphs[subgraph_name].nodes[node_id] = node

            # edges
            for graphviz_edge in G.edges():
                node1_id = graphviz_edge[0]
                node2_id = graphviz_edge[1]
                subgraph_name_1 = self.find_subgraph_for_node(node1_id)
                subgraph_name_2 = self.find_subgraph_for_node(node2_id)
                if selected_subgraphs:
                    if subgraph_name_1 in selected_subgraphs and subgraph_name_2 in selected_subgraphs:
                        edge = Edge(self.subgraphs[subgraph_name_1].nodes[node1_id], self.subgraphs[subgraph_name_2].nodes[node2_id],directed=self.directed)
                        if subgraph_name_1 == subgraph_name_2:
                            self.subgraphs[subgraph_name_1].edges[(node1_id, node2_id)] = edge
                        else:
                            self.edges[(node1_id, node2_id)] = edge
                else:
                    edge = Edge(self.subgraphs[subgraph_name_1].nodes[node1_id], self.subgraphs[subgraph_name_2].nodes[node2_id], directed=self.directed)
                    self.edges[(node1_id, node2_id)] = edge

        else:
            # nodes
            for i, graphviz_node in enumerate(G.nodes()):
                node = Node(_id=graphviz_node.get_name(),number=i,colour=colour)
                self.nodes[node.id] = node

            # edges
            for graphviz_edge in G.edges():
                node1_id = graphviz_edge[0]
                node2_id = graphviz_edge[1]

                weight = graphviz_edge.attr[weight_name] if weight_name in graphviz_edge.attr else None

                self.nodes[node1_id].add_out_neighbour(self.nodes[node2_id])
                self.nodes[node2_id].add_in_neighbour(self.nodes[node1_id])
                if self.directed:
                    edge = Edge(self.nodes[node1_id], self.nodes[node2_id], weight=weight, directed=True)
                else:
                    edge = Edge(self.nodes[node1_id], self.nodes[node2_id], weight=weight)
                self.edges[(node1_id, node2_id)] = edge

    def return_fig(self,labels=False,axis=False,subgraphs=False, title=None, bundled=False, zorder=1, draw_edges=True):
        print("Updating Figure")
        if subgraphs:
            for name, subgraph in self.subgraphs.items():
                subgraph.return_fig(title=name)
        else:

            node_radius = min(1.2,(self.min_max_x[1] - self.min_max_x[0]) / (5 * math.sqrt(len(self.nodes))))
            x_lim = self.min_max_x + np.array([-node_radius,node_radius])
            y_lim = self.min_max_y + np.array([-node_radius,node_radius])
            self.ax.set_xlim(x_lim)
            self.ax.set_ylim(y_lim)
            self.ax.set_title(title)
            self.ax.set_aspect('equal', adjustable='box')
            plt.axis(axis)

            if draw_edges:
                for edge in self.edges.values():
                    if bundled:
                        edge.get_fig_coordinates(self.ax,self.ax)
                    else:
                        edge.update_line()
                        edge.line.zorder = -zorder # Ensure Edges are behind Nodes
                        self.ax.add_patch(edge.line)


            for node in self.nodes.values():
                node.circle.radius = node_radius
                node.circle.zorder = zorder # Ensure Nodes are in front of Edges
                self.ax.add_patch(node.circle)
                if labels:
                    node.show_label(self.ax)

            self.fig.canvas.draw()
            if bundled:
                bundled_edges = edge_bundling_precomputed(self,C = 4, I = 20 ,s = .1, n0 = 2, kP =.01)

                for edge, control_points in bundled_edges.items():

                    for i in range(len(control_points) - 1):
                        start_point = control_points[i]
                        end_point = control_points[i + 1]

                        # Create a connection patch between the start and end points
                        conn_patch = ConnectionPatch(start_point, end_point, "figure fraction", "figure fraction",
                                                    arrowstyle="-", shrinkA=0, shrinkB=0,
                                                    mutation_scale=10, fc="w",lw=0.2, color="grey")
                        self.fig.add_artist(conn_patch)
                #######



            return self.fig

    ## STEP 1
    def random_layout(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0), subgraphs=False):
        """
        Positions the nodes at random locations within specified ranges.

        Args:
        x_range (tuple, optional): The range of x coordinates. Defaults to (0.0, 1.0).
        y_range (tuple, optional): The range of y coordinates. Defaults to (0.0, 1.0).
        subgraphs (bool, optional): If True, applies the random layout to each subgraph. Defaults to False.
        """
        # Apply layout to subgraphs if specified
        if subgraphs:
            for subgraph in self.subgraphs.values():
                subgraph.random_layout()

        # Position each node at a random location within the specified ranges
        for node in self.nodes.values():
            x = np.random.uniform(x_range[0], x_range[1])  # Random x coordinate
            y = np.random.uniform(y_range[0], y_range[1])  # Random y coordinate
            node.circle.center = np.array([x, y])  # Set node position

    def circular_layout(self, center=(.5, .5), radius=.5, subgraphs=False):
        """
        Arranges the nodes in a circular layout centered at a specified point.

        Args:
            center (tuple, optional): The (x, y) coordinates for the center of the circle. Defaults to (.5, .5).
            radius (float, optional): The radius of the circle. Defaults to .5.
            subgraphs (bool, optional): If True, applies the circular layout to each subgraph. Defaults to False.
        """
        # Apply layout to subgraphs if specified
        if subgraphs:
            for subgraph in self.subgraphs.values():
                subgraph.circular_layout()

        # Calculate the position of each node to distribute them evenly in a circle
        N = len(self.nodes)  # Total number of nodes
        cx, cy = center  # Center coordinates
        for i, node in enumerate(self.nodes.values()):
            angle = 2 * np.pi * i / N  # Angle for the current node
            x = cx + radius * np.cos(angle)  # X coordinate
            y = cy + radius * np.sin(angle)  # Y coordinate
            node.circle.center = np.array([x, y])  # Set node position

    ## STEP 2 Graph Traversals
    def bfs(self, root):
        """
        Performs a breadth-first search (BFS) starting from a specified root node.

        Args:
            root: The identifier for the root node from which the BFS starts.
        """
        visited = set()  # Keep track of visited nodes
        self.bfs_order = []  # Order of nodes visited in BFS
        self.bfs_tree = {root: []}  # Tree resulting from the BFS
        self.bfs_non_tree_edges = {root: []}  # Edges not in the BFS tree
        queue = deque([root])  # Queue for BFS
        visited.add(root)

        while queue:
            node_id = queue.popleft()
            for neighbour, _ in self.nodes[node_id].out_neighbours:
                if neighbour.id not in visited:
                    visited.add(neighbour.id)
                    queue.append(neighbour.id)

                    # Handling non-tree edges
                    for non_tree_neighbour, _ in self.nodes[node_id].out_neighbours + self.nodes[node_id].in_neighbours:
                        if non_tree_neighbour != neighbour:
                            self.bfs_non_tree_edges[node_id].append(non_tree_neighbour.id)

                    # Update BFS tree and order
                    if neighbour.id not in self.bfs_tree:
                        self.bfs_tree[neighbour.id] = []
                        self.bfs_non_tree_edges[neighbour.id] = []
                    self.bfs_tree[node_id].append(neighbour.id)
                    self.bfs_order.append(neighbour.id)

    def dfs(self, root):
        """
        Performs a depth-first search (DFS) starting from a specified root node.

        Args:
            root: The identifier for the root node from which the DFS starts.
        """
        visited = set()  # Keep track of visited nodes
        self.dfs_order = []  # Order of nodes visited in DFS
        self.dfs_tree = {root: []}  # Tree resulting from the DFS
        self.dfs_non_tree_edges = {node_id: [] for node_id in self.nodes.keys()}  # Edges not in the DFS tree

        def dfs_recursive(node_id):
            """
            Recursively visits nodes in a depth-first manner.

            Args:
                node_id: The identifier of the current node being visited.
            """
            visited.add(node_id)
            self.dfs_order.append(node_id)
            for neighbour, _ in self.nodes[node_id].out_neighbours:
                if neighbour.id not in visited:
                    # Node not visited, add to DFS tree and continue recursion
                    self.dfs_tree[node_id].append(neighbour.id)
                    self.dfs_tree[neighbour.id] = []
                    dfs_recursive(neighbour.id)
                else:
                    # Node already visited, add to non-tree edges
                    self.dfs_non_tree_edges[node_id].append(neighbour.id)

        dfs_recursive(root)  # Start DFS from root

    # STEP 3 FORCE DIRECTED GRAPHS
    def repulsive_forces(self, k):  #k is C_rep
        forces = {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()}
        for u_id, u in self.nodes.items():
            for v in self.nodes.values():
                if u != v and not u.has_neighbour(v):
                    delta = np.array(u.circle.center) - np.array(v.circle.center)
                    distance = np.linalg.norm(delta)
                    if distance > 0:
                        repulsive_force = k / distance**2
                        forces[u_id] += repulsive_force * (delta / distance) #delta is normalized
        return forces

    def spring_forces(self, k, optimal_length): #k is C_spring

        forces = {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()}

        for u_id, u in self.nodes.items():
            for v in self.nodes.values():
                if u.has_neighbour(v):
                    delta = np.array(u.circle.center) - np.array(v.circle.center)
                    distance = np.linalg.norm(delta)
                    if distance > 0:
                        spring_force_magnitude = k * np.log(distance / optimal_length)
                        spring_force = spring_force_magnitude * (delta / distance)
                        forces[u_id] += spring_force

        return forces

    def spring_embedder(self, k_rep=1, k_spring=2, optimal_length=.1, iterations=100, threshold=1e-5, delta=.001, subgraphs=False):
        if subgraphs:
            for sg in self.subgraphs:
                sg.spring_embedder(k_rep=k_rep, k_spring=k_spring, optimal_length=optimal_length, iterations=iterations, threshold=threshold,delta=delta)

        else:
            t = 1
            while t <= iterations:
                rep_forces = self.repulsive_forces(k_rep)
                spr_forces = self.spring_forces(k_spring, optimal_length)

                net_force = {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()}  # Initialize net forces

                for node_id, node in self.nodes.items():
                    net_force[node_id] = rep_forces[node_id] + spr_forces[node_id]

                for node_id, node in self.nodes.items():
                    self.nodes[node_id].circle.center += delta * net_force[node_id]

                    self.min_max_x = np.array([min(node.circle.center[0],self.min_max_x[0]),
                                                max(node.circle.center[0],self.min_max_x[1])])
                    self.min_max_y = np.array([min(node.circle.center[1],self.min_max_y[0]),
                                                max(node.circle.center[1],self.min_max_y[1])])

                max_force = max(np.linalg.norm(force) for force in net_force.values())
                if max_force < threshold:
                    break

                t += 1

    def repulsive_forces_1(self, ideal_edge_length):
        forces = {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()}
        for u_id,u in self.nodes.items():
            for v in self.nodes.values():
                if u != v:  # Exclude self-repulsion
                    delta = np.array(u.circle.center) - np.array(v.circle.center)
                    distance = np.linalg.norm(delta) + 1e-6  # Prevent division by zero
                    repulsive_force = ideal_edge_length**2 / distance
                    forces[u_id] += repulsive_force * (delta / distance)
        return forces

    def attractive_forces_1(self, ideal_edge_length, mass_bool=False):
        forces = {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()}
        if mass_bool:
            masses = {node_id: 1 + node.degree() / 2 for node_id,node in self.nodes.items()}  # Calculate node mass**1

        for u_id, v_id in self.edges.keys():  # Only iterate over edges
            delta = np.array(self.nodes[v_id].circle.center) - np.array(self.nodes[u_id].circle.center)
            distance = np.linalg.norm(delta) + 1e-6  # Prevent division by zero
            spring_force_magnitude = (distance**2 / ideal_edge_length) / masses[u_id] if mass_bool else distance**2 / ideal_edge_length
            # Feedback: Instead of updating the forces array, store the force in a separate variable
            force = spring_force_magnitude * (delta / distance)
            forces[u_id] += force
            forces[v_id] -= force # Apply equal and opposite force
        return forces

    def calculate_spring_forces(self, rep_forces, attr_forces):
        spring_forces = {node_id: rep_forces[node_id] for node_id in self.nodes.keys()}
        for u_id, v_id in self.edges.keys():
            spring_forces[u_id] += attr_forces[u_id]
            spring_forces[v_id] += attr_forces[v_id]
        return spring_forces

    def magnetic_forces(self, magnetic_constant):
        forces = {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()}
        field_direction = np.array([0, 1])  # use [1, 0] for a horizontal field

        for u, v in self.edges.keys():
            delta = np.array(self.nodes[v].circle.center) - np.array(self.nodes[u].circle.center)
            edge_direction = delta / (np.linalg.norm(delta) + 1e-6)
            cos_theta = np.dot(edge_direction, field_direction)
            angle = np.arccos(np.clip(cos_theta, -1, 1))
            magnetic_force_magnitude = magnetic_constant * (1 - cos_theta)
            # Calculate perpendicular direction to the edge direction to apply force
            perpendicular_dir = np.array([-edge_direction[1], edge_direction[0]])
            magnetic_force = magnetic_force_magnitude * perpendicular_dir

            forces[u] += magnetic_force
            forces[v] -= magnetic_force
        return forces

    def gravitational_forces(self, center_point, gravitational_constant):
        forces = {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()}
        for node_id, node in self.nodes.items():
            delta = center_point - np.array(node.circle.center)

            distance = np.linalg.norm(delta)
            grav_force = gravitational_constant * delta
            forces[node_id] += grav_force / distance if distance > 0 else 0
        return forces

    def spring_embedder_f(self, ideal_length=.1, K=100, epsilon=1e-4, delta=.1, gravitational_constant=None, magnetic_constant=None):
        if self.subgraphs:

            for subgraph in self.subgraphs.values():
                subgraph.spring_embedder_f()

        else:
            center_point = np.array([0.5, 0.5])
            t = 0
            cooling_factor = 0.95
            min_delta = 0.00001

            while t < K:
                rep_forces = self.repulsive_forces_1(ideal_length)
                attr_forces = self.attractive_forces_1(ideal_length)
                grav_forces = self.gravitational_forces(center_point, gravitational_constant) if gravitational_constant else {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()} #Gravity
                mag_forces = self.magnetic_forces(magnetic_constant) if magnetic_constant else {node_id: np.array([0.0, 0.0]) for node_id in self.nodes.keys()} #magnetic



                for node_id, node in self.nodes.items():
                    net_force = rep_forces[node_id] + attr_forces[node_id] + grav_forces[node_id] + mag_forces[node_id] #+gravity/magnetic

                    displacement = np.clip(delta * net_force, -delta, delta)
                    node.circle.center += displacement
                    self.min_max_x = np.array([min(node.circle.center[0],self.min_max_x[0]),
                                                max(node.circle.center[0],self.min_max_x[1])])
                    self.min_max_y = np.array([min(node.circle.center[1],self.min_max_y[0]),
                                                max(node.circle.center[1],self.min_max_y[1])])

                delta = max(min_delta, cooling_factor * delta)

                max_force = max(np.linalg.norm(node.circle.center) for node in self.nodes.values())
                if max_force < epsilon:
                    break

                t += 1

    ## STEP 5 Subplots
    def find_subgraph_containing_node(self, node_id):
        for name, subgraph in self.subgraphs.items():
            if node_id in subgraph.nodes:
                return name
        return None

    def find_subgraph_for_node(self, node_id):
        for subgraph_name, subgraph in self.subgraphs.items():
            if node_id in subgraph.nodes:
                return subgraph_name

    def update_ax(self, labels=True, axis=False, title=None, ax=None,bundled=False):
        print("Updating Figure")
        # Dynamically adjust node radius based on the subplot size and number of nodes
        node_radius = min(.05, (self.min_max_x[1] - self.min_max_x[0]) / np.sqrt(len(self.nodes)) * 0.2)

        # Ensure nodes are placed within the subplot area
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
        if not bundled:
            for edge in self.edges.values():
                edge.update_line()
                ax.add_patch(edge.line)

        for node in self.nodes.values():
            node.circle.radius = node_radius
            ax.add_patch(node.circle)
            min_x, min_y = min(min_x, node.circle.center[0]), min(min_y, node.circle.center[1])
            max_x, max_y = max(max_x, node.circle.center[0]), max(max_y, node.circle.center[1])
            if labels:
                node.show_label(ax)

        # Add a buffer around the bounding box to ensure it fits within the subplot
        buffer = node_radius * 2
        ax.set_xlim(min_x - buffer, max_x + buffer)
        ax.set_ylim(min_y - buffer, max_y + buffer)
        ax.set_title(title)
        plt.axis(axis)

    def return_subplots(self, labels=True, axis=True, title=None, figsize=(15, 15),bundled=(False,False)):
        num_subgraphs = len(self.subgraphs)
        num_cols = int(np.ceil(np.sqrt(num_subgraphs)))  # Columns based on square root of number of subgraphs
        num_rows = max((num_subgraphs + num_cols - 1) // num_cols, 1)  # Ensure at least 1 row

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=False)
        if num_subgraphs > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]  # Ensure axes_flat is always iterable

        for ax in axes_flat[num_subgraphs:]:  # Hide unused subplots
            ax.set_visible(False)


        for i, (name, subgraph) in enumerate(self.subgraphs.items()):
            ax = axes_flat[i]
            subgraph.update_ax(labels=labels, axis=axis, title=name, ax=ax,bundled=bundled[1])
            ax.set_aspect('equal', adjustable='box')

        fig.canvas.draw()

        for edge in self.edges.values():
            node1_id, node2_id = edge.node1.id, edge.node2.id
            subgraph1 = self.find_subgraph_containing_node(node1_id)
            subgraph2 = self.find_subgraph_containing_node(node2_id)

            # Determine the subplot axes for each node
            ax1 = axes_flat[list(self.subgraphs.keys()).index(subgraph1)]
            ax2 = axes_flat[list(self.subgraphs.keys()).index(subgraph2)]

            fig_coor = edge.get_fig_coordinates(ax1=ax1,ax2=ax2)

            if not bundled[0]:
                fig.add_artist(ConnectionPatch(fig_coor[0], fig_coor[1], coordsA='figure fraction', coordsB='figure fraction', lw=0.2, color="blue"))

        if bundled[1]:
            for sg_name, subgraph in self.subgraphs.items():
                ax = axes_flat[list(self.subgraphs.keys()).index(sg_name)]
                for edge in subgraph.edges.values():
                    fig_coor = edge.get_fig_coordinates(ax1=ax,ax2=ax)

                bundled_edges = edge_bundling(subgraph,C = 5, I = 25 ,s = 0.04, n0 = 2, kP = 0.1 )

                for edge, control_points in bundled_edges.items():

                    for i in range(len(control_points) - 1):
                        start_point = control_points[i]
                        end_point = control_points[i + 1]

                        # Create a connection patch between the start and end points
                        conn_patch = ConnectionPatch(start_point, end_point, "figure fraction", "figure fraction",
                                                    arrowstyle="-", shrinkA=0, shrinkB=0,
                                                    mutation_scale=10, fc="w",lw=0.3, color="grey",zorder=.1)
                        fig.add_artist(conn_patch)

        if bundled[0]:
            bundled_edges = edge_bundling_precomputed(self,C = 6, I = 30 ,s = .1, n0 = 2, kP =.01)

            for edge, control_points in bundled_edges.items():

                for i in range(len(control_points) - 1):
                    start_point = control_points[i]
                    end_point = control_points[i + 1]

                    # Create a connection patch between the start and end points
                    conn_patch = ConnectionPatch(start_point, end_point, "figure fraction", "figure fraction",
                                                arrowstyle="-", shrinkA=0, shrinkB=0,
                                                mutation_scale=10, fc="w",lw=0.2, color="blue")
                    fig.add_artist(conn_patch)


        plt.suptitle(title)
        return fig

    # STEP 6 PROJECTIONS FOR GRAPHS
    def distances_matrix(self):
        """
        Calculates the distances matrix for all pairs of nodes in the graph,
        using the Floyd-Warshall algorithm to find the shortest paths.

        Returns:
            np.ndarray: A 2D array of distances between all pairs of nodes.
        """
        N = len(self.nodes)
        D = np.ones((N,N)) * 10000 # Start with a high value representing 'infinite' distance

        numbered_nodes = {node.number:node for node in self.nodes.values()} # Map nodes to their numbers for easier access
        numbered_edges = {(edge.node1.number, edge.node2.number): edge for edge in self.edges.values()}
        for (i,j), edge in numbered_edges.items():
            D[i,j] = int(edge.weight) if edge.weight else 1 # Use weight if available, else default to 1
            D[j, i] = D[i,j] # symmetry!

        # Set the diagonal to 0 since the distance from a node to itself is always 0
        for i in range(N):
            D[i, i] = 0

        # Floyd-Warshall algorithm: Update the distances matrix with shorter paths found via intermediate nodes
        for k, knode in numbered_nodes.items():
            for i, inode in numbered_nodes.items():
                for j, jnode in numbered_nodes.items():
                    # If a shorter path is found via node k, update the distances
                    if D[i,j] > D[i,k] + D[k,j]:
                        D[i,j] = D[i,k] + D[k,j]
                        D[j, i] = D[i, j] # symmetry!

        self.D = D
        return D

    def mds_coordinates(self,random_state=7):
        from sklearn.manifold import MDS
        D = self.distances_matrix()
        mds = MDS(n_components=2, dissimilarity='precomputed',random_state=random_state)
        Y = mds.fit_transform(D)
        # Sorting and iterating through nodes based on Node.number
        for i, node in enumerate(sorted(self.nodes.values(), key=lambda val: val.number)):

            node.circle.center =  Y[i]
            self.min_max_x = np.array([min(node.circle.center[0],self.min_max_x[0]),
                                        max(node.circle.center[0],self.min_max_x[1])])
            self.min_max_y = np.array([min(node.circle.center[1],self.min_max_y[0]),
                                        max(node.circle.center[1],self.min_max_y[1])])

    def tsne_coordinates(self, perplexity = 7, random_state=0):
        from sklearn.manifold import TSNE
        D = self.D
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=1, metric='precomputed',random_state=random_state)
        Y = tsne.fit_transform(D)

        # Sorting and iterating through nodes based on Node.number
        for i, node in enumerate(sorted(self.nodes.values(), key=lambda val: val.number)):

            node.circle.center =  Y[i]
            self.min_max_x = np.array([min(node.circle.center[0],self.min_max_x[0]),
                                        max(node.circle.center[0],self.min_max_x[1])])
            self.min_max_y = np.array([min(node.circle.center[1],self.min_max_y[0]),
                                        max(node.circle.center[1],self.min_max_y[1])])

    def isomap_coordinates(self, n_neighbours = 10):
        from sklearn.manifold import Isomap
        D = self.D
        iso = Isomap(n_neighbors=n_neighbours, n_components=2,metric='precomputed')
        Y = iso.fit_transform(D)

        # Sorting and iterating through nodes based on Node.number
        for i, node in enumerate(sorted(self.nodes.values(), key=lambda val: val.number)):

            node.circle.center =  Y[i]
            self.min_max_x = np.array([min(node.circle.center[0],self.min_max_x[0]),
                                        max(node.circle.center[0],self.min_max_x[1])])
            self.min_max_y = np.array([min(node.circle.center[1],self.min_max_y[0]),
                                        max(node.circle.center[1],self.min_max_y[1])])
