import pygraphviz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import math
import networkx as nx

from classes_clean.node import Node
from classes_clean.edge import Edge

class Graph:
    def __init__(self, incoming_dot_file=None, directed = False, subgraphs = False, a_subgraph = False, selected_subgraphs=None,weight_name="weight"):
        # directed graph
        self.directed = directed
        self.feedback_set = set()
        # graph nodes and edges
        self.nodes, self.edges = {}, {}
        self.subgraphs = None
        if incoming_dot_file:
            self.load_graph(incoming_dot_file, subgraphs=subgraphs, selected_subgraphs=selected_subgraphs, weight_name=weight_name)

        # graph figure for visualisation
        self.fig = None if a_subgraph or subgraphs else plt.figure(figsize=(7,7))
        self.ax =  None if a_subgraph or subgraphs else self.fig.gca()
        self.axes = []
        self.min_max_x = np.array([0,1])
        self.min_max_y = np.array([0,1])

        # graph traversals
        self.dfs_order = []
        self.bfs_order= []
        # trees
        self.dfs_tree = {}
        self.bfs_tree = {}

        self.bfs_non_tree_edges = None
        self.dfs_non_tree_edges = None


    def load_graph(self, dot_file_path, subgraphs=False, selected_subgraphs=None,weight_name="weight"):
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
                            node = Node(_id=node_id,number=i)
                            self.subgraphs[subgraph_name].nodes[node_id] = node
                else:
                    subgraph_name = subgraph.name
                    self.subgraphs[subgraph_name] = Graph(a_subgraph=True)
                    # nodes
                    for graphviz_node in subgraph.nodes():
                        node_id = graphviz_node.get_name()
                        node = Node(_id=node_id)
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
                        self.edges[(node1_id, node2_id)] = edge
                else:
                    edge = Edge(self.subgraphs[subgraph_name_1].nodes[node1_id], self.subgraphs[subgraph_name_2].nodes[node2_id], directed=self.directed)
                    self.edges[(node1_id, node2_id)] = edge

        else:
            # nodes
            for i, graphviz_node in enumerate(G.nodes()):
                node = Node(_id=graphviz_node.get_name(),number=i)
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

    def find_subgraph_for_node(self, node_id):
        for subgraph_name, subgraph in self.subgraphs.items():
            if node_id in subgraph.nodes:
                return subgraph_name

    def add_node(self, node_id):
        self.nodes[node_id] = Node(node_id)

    def return_fig(self,labels=True,axis=False,subgraphs=False, title=None):
        print("Updating Figure")
        if subgraphs:
            for name, subgraph in self.subgraphs.items():
                subgraph.return_fig(title=name)
        else:
            node_radius = min(.1,(self.min_max_x[1] - self.min_max_x[0]) / (5 * math.sqrt(len(self.nodes))))

            for edge in self.edges.values():
                edge.update_line()
                self.ax.add_patch(edge.line)

            for node in self.nodes.values():
                #print(node.circle.center)
                node.circle.radius = node_radius
                self.ax.add_patch(node.circle)
                if labels:
                    node.show_label(self.ax)


            x_lim = self.min_max_x + np.array([-node_radius,node_radius])
            y_lim = self.min_max_y + np.array([-node_radius,node_radius])
            self.ax.set_xlim(x_lim)
            self.ax.set_ylim(y_lim)
            self.ax.set_title(title)
            plt.axis(axis)

            return self.fig

    def update_ax(self, labels=True, axis=False, title=None, ax = None):
        print("Updating Figure")
        node_radius = min(.1, (self.min_max_x[1] - self.min_max_x[0]) / (5 * math.sqrt(len(self.nodes))))

        print(len(self.nodes), len(self.edges))
        for node in self.nodes.values():
            node.circle.radius = node_radius
            ax.add_patch(node.circle)
            if labels:
                node.show_label(ax)

        x_lim = self.min_max_x + np.array([-(node_radius+.1), node_radius+.1])
        y_lim = self.min_max_y + np.array([-(node_radius+.1), node_radius+.1])
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(title)

        ax.axis(axis)

    def return_subplots(self, labels=True, axis=False, title=None):
        num_subgraphs = len(self.subgraphs)
        num_cols = int(np.ceil(np.sqrt(num_subgraphs)))  # Ensure at least 1 column
        num_rows = (num_subgraphs + num_cols - 1) // num_cols  # Ensure at least 1 row

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex='col', sharey='row')
        # Flatten axes array if multiple axes, else wrap in a list for consistent handling
        axes_flat = axes.flatten() if num_subgraphs > 1 else [axes]

        for ax in axes_flat[num_subgraphs:]:  # Hide unused subplots
            ax.set_visible(False)

        for (name, subgraph), ax in zip(self.subgraphs.items(), axes_flat):
            subgraph.update_ax(labels=labels, axis=axis, title=name, ax=ax)
            ax.set_aspect('equal')
            # Draw bounding box around the subgraph
            min_x, max_x = ax.get_xlim()
            min_y, max_y = ax.get_ylim()
            bbox = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=2, edgecolor="green", facecolor='none')
            ax.add_patch(bbox)

            for edge in self.edges.values():
                node1_id, node2_id = edge.node1.id, edge.node2.id
                subgraph1 = self.find_subgraph_containing_node(node1_id)
                subgraph2 = self.find_subgraph_containing_node(node2_id)
                if subgraph1 is not None and subgraph2 is not None and subgraph1 != subgraph2:
                    # Draw connection between subgraphs
                    ax1 = axes_flat[list(self.subgraphs.keys()).index(subgraph1)]
                    ax2 = axes_flat[list(self.subgraphs.keys()).index(subgraph2)]
                    edge.update_line(ax1=ax1,ax2=ax2,color="red",coordsA='data',coordsB='data')
                    ax.add_patch(edge.line)
                else:
                    edge.update_line()
                    ax=axes_flat[list(self.subgraphs.keys()).index(subgraph1)]
                    ax.add_patch(edge.line)

        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

    def find_subgraph_containing_node(self, node_id):
        for name, subgraph in self.subgraphs.items():
            if node_id in subgraph.nodes:
                return name
        return None

    def circular_layout(self, center=(.5,.5), radius=.5, subgraphs=False):
        print("CIRCULAR LAYOUT")
        if subgraphs:
            for subgraph in self.subgraphs.values():
                subgraph.circular_layout()

        N = len(self.nodes)
        cx, cy = center
        i = 0
        for node in self.nodes.values():
            angle = 2* np.pi * i / N
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            node.circle.center = np.array([x,y])
            i += 1

    def random_layout(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0), subgraphs=False):
        print("RANDOM LAYOUT")
        if subgraphs:
            for subgraph in self.subgraphs.values():
                subgraph.random_layout()
        for node in self.nodes.values():
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            node.circle.center = np.array([x,y])

    def dfs(self, root):
        visited = set()
        self.dfs_order = []
        self.dfs_tree = {root: []}
        self.dfs_non_tree_edges = {node_id:[] for node_id in self.nodes.keys()}
        def dfs_recursive(node_id):
            visited.add(node_id)
            self.dfs_order.append(node_id)
            for neighbour, _ in self.nodes[node_id].out_neighbours:
                if neighbour.id not in visited:
                    self.dfs_tree[node_id].append(neighbour.id)
                    self.dfs_tree[neighbour.id] = []
                    dfs_recursive(neighbour.id)
                else:
                    self.dfs_non_tree_edges[node_id].append(neighbour.id)

        dfs_recursive(root)

    def bfs(self, root):
        visited = set()
        self.bfs_order = []
        self.bfs_tree = {root: []}
        self.bfs_non_tree_edges = {root:[]}
        queue = deque([root])
        visited.add(root)
        while queue:
            node_id = queue.popleft()
            for neighbour,_ in self.nodes[node_id].out_neighbours:
                if neighbour.id not in visited:
                    visited.add(neighbour.id)
                    queue.append(neighbour.id)

                    #
                    for non_tree_neighbour,_ in self.nodes[node_id].out_neighbours + self.nodes[node_id].in_neighbours:
                        if non_tree_neighbour != neighbour:
                            self.bfs_non_tree_edges[node_id].append(non_tree_neighbour.id)

                    if neighbour.id not in self.bfs_tree:
                        self.bfs_tree[neighbour.id] = []
                        #
                        self.bfs_non_tree_edges[neighbour.id] = []
                    self.bfs_tree[node_id].append(neighbour.id)
                    self.bfs_order.append(neighbour.id)

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

    def spring_embedder(self, k_rep=1, k_spring=2, optimal_length=1, iterations=100, threshold=1e-5, delta=.001, subgraphs=False):
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
            print(self.nodes)
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

    def distances_matrix(self):
        N = len(self.nodes)
        D = np.ones((N,N)) * 10000
        numbered_nodes = {node.number:node for node in self.nodes.values()}
        numbered_edges = {(edge.node1.number, edge.node2.number): edge for edge in self.edges.values()}
        for (i,j), edge in numbered_edges.items():
            D[i,j] = int(edge.weight) if edge.weight else 1 # if no weights give path vlalue
            D[j, i] = D[i,j] # symmetry!

        #self loop
        for i in range(N):
            D[i, i] = 0

        for k, knode in numbered_nodes.items():
            for i, inode in numbered_nodes.items():
                for j, jnode in numbered_nodes.items():
                    if D[i,j] > D[i,k] + D[k,j]:
                        D[i,j] = D[i,k] + D[k,j]
                        D[j, i] = D[i, j] # symmetry!
        return D
