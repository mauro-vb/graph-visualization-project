import pygraphviz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import math

from classes_clean.node import Node
from classes_clean.edge import Edge

class Graph:
    def __init__(self, incoming_dot_file=None, directed = False, subgraphs = False, a_subgraph = False, selected_subgraphs=None,weight_name="weight"):
        # directed graph
        self.directed = directed
        self.feedback_set = set()
        # graph nodes and edges
        self.nodes, self.edges = {}, {}
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

    def return_fig(self,labels=True,axis=True,subgraphs=False, title=None):
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
        num_cols = int(np.ceil(np.sqrt(num_subgraphs)))  # Number of columns based on square root of num_subgraphs

        # Adjust num_rows to ensure that num_cols * num_rows is greater than or equal to num_subgraphs
        num_rows = (num_subgraphs + num_cols - 1) // num_cols

        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)

        for (name, subgraph), ax_sub in zip(self.subgraphs.items(), self.axes.flatten()):
            subgraph.update_ax(labels=labels, axis=axis, title=name, ax=ax_sub)
            ax_sub.set_aspect('equal')
             # Draw bounding box around the subgraph
            min_x, max_x = ax_sub.get_xlim()
            min_y, max_y = ax_sub.get_ylim()
            bbox = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=2, edgecolor="green", facecolor='none')
            ax_sub.add_patch(bbox)

            for edge in self.edges.values():
                node1_id, node2_id = edge.node1.id, edge.node2.id
                subgraph1 = self.find_subgraph_containing_node(node1_id)
                subgraph2 = self.find_subgraph_containing_node(node2_id)
                if subgraph1 is not None and subgraph2 is not None and subgraph1 != subgraph2:
                    # Draw connection between subgraphs
                    ax1 = self.axes.flatten()[list(self.subgraphs.keys()).index(subgraph1)]
                    ax2 = self.axes.flatten()[list(self.subgraphs.keys()).index(subgraph2)]
                    edge.update_line(ax1=ax1,ax2=ax2,color="red",coordsA='data',coordsB='data')
                    ax_sub.add_patch(edge.line)
                else:
                    edge.update_line()
                    ax=self.axes.flatten()[list(self.subgraphs.keys()).index(subgraph1)]
                    ax.add_patch(edge.line)
        #plt.tight_layout()
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
        def dfs_recursive(node_id):
            visited.add(node_id)
            self.dfs_order.append(node_id)
            for neighbour, _ in self.nodes[node_id].out_neighbours:
                if neighbour.id not in visited:
                    self.dfs_tree[node_id].append(neighbour.id)
                    self.dfs_tree[neighbour.id] = []
                    dfs_recursive(neighbour.id)
        dfs_recursive(root)

    def bfs(self, root):
        visited = set()
        self.bfs_order = []
        self.bfs_tree = {root: []}
        queue = deque([root])

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            self.bfs_order.append(node_id)

            for neighbour,_ in self.nodes[node_id].out_neighbours:
                if neighbour.id not in visited:
                    self.bfs_tree[neighbour.id] = []
                    self.bfs_tree[node_id].append(neighbour.id)
                    #self.bfs_tree[neighbour.id] = []
                    queue.append(neighbour.id)

    def force_directed_graph(self, embedder_type="Eades", K=500, epsilon=1e-4, delta=.1, c=.9,c_rep=1,c_spring=2, subgraphs=False):
        print("force directed graph computation")
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

        if subgraphs:
            for subgraph in self.subgraphs.values():
                    subgraph.force_directed_graph()

        for _ in range(K):
            displacement = {v: np.zeros(2) for v in self.nodes.keys()}

            # calculate repulsive forces
            for key_u, u in self.nodes.items():
                for v in self.nodes.values():
                    diff = u.circle.center - v.circle.center
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        disp = repulsive_force(distance, diff)
                        displacement[key_u] += disp

            # calculate attractive forces
            for key_u, u in self.nodes.items():
                for key_v, v in self.nodes.items():
                    diff = u.circle.center - v.circle.center
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        disp = attractive_force(distance, diff)
                        displacement[key_u] -= disp
                        displacement[key_v] += disp

            # update positions
            for key_v, v in self.nodes.items():
                length = np.linalg.norm(displacement[key_v])
                if length > 0:
                    # displacement vector is normalized
                    v.circle.center += delta * displacement[key_v] / length
                    self.min_max_x = np.array([min(v.circle.center[0],self.min_max_x[0]),
                                               max(v.circle.center[0],self.min_max_x[1])])
                    self.min_max_y = np.array([min(v.circle.center[1],self.min_max_y[0]),
                                               max(v.circle.center[1],self.min_max_y[1])])

            max_displacement = max(np.linalg.norm(disp) for disp in displacement.values()) if displacement.values() else 0

            if max_displacement < epsilon:
                break

    def remove_cycles(self):
        order = []
        self.feedback_set = set()
        for node_id in sorted(self.nodes.keys(), key=lambda x: int(x)):
            node = self.nodes[node_id]
            order.append(node_id)
            for neighbour in node.out_neighbours:
                if neighbour.id in order:
                    # inverse edge direction in node
                    node.out_neighbours.remove(neighbour)
                    node.in_neighbours.append(neighbour)
                    # inverse edge direction in neighbour
                    neighbour.in_neighbours.remove(node)
                    neighbour.out_neighbours.append(node)
                    # inverse edge
                    self.edges[(node_id,neighbour.id)].invert()
                    self.feedback_set.add(self.edges[(node_id,neighbour.id)])

    def heuristic_with_guarantees(self):
        edges_to_reverse = set()

        while len(self.nodes) > 0:
            sinks = [node for node in self.nodes.values() if len(node.in_neighbours) > 0 and len(node.out_neighbours) == 0]
            for sink in sinks:
                for in_n in sink.in_neighbours:
                    edges_to_reverse.add((in_n, sink))
                self.remove_node(sink)

            isolated_nodes = [isolate for isolate in self.nodes.values() if len(isolate.in_neighbours) + len(isolate.out_neighbours) == 0]
            for isolate in isolated_nodes:
                self.remove_node(isolate)

            sources = [node for node in self.nodes.values() if len(node.out_neighbours) > 0 and len(node.in_neighbours) == 0]
            for source in sources:
                for out_n in source.out_neighbours:
                    edges_to_reverse.add((out_n, source))
                self.remove_node(source)

            # If graph is non-empty, select a node with max |N_outgoing| - |N_incoming|
            if len(self.nodes) > 0:
                node = max(self.nodes.values(), key=lambda node: len(node.in_neighbours) - len(node.out_neighbours))
                for out_n in node.out_neighbours:
                    edges_to_reverse.add((out_n, node))
                self.remove_node(node)

        return edges_to_reverse

    def remove_node(self, node):
        for neighbour in node.out_neighbours:
            neighbour.in_neighbours.remove(node)
        for neighbour in node.in_neighbours:
            neighbour.out_neighbours.remove(node)
        del self.nodes[node.id]

    def has_cycle(self):
        visited = set()
        recursion_stack = set()

        def dfs(node_id):
            if node_id in recursion_stack:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            recursion_stack.add(node_id)
            for neighbour in self.nodes[node_id].out_neighbours:
                if dfs(neighbour.id):
                    return True
            recursion_stack.remove(node_id)
            return False

        for node in self.nodes:
            if dfs(node):
                return True
        return False

    def topological_sort(self):
        # Dictionary to store in-degrees of nodes
        in_degrees = {node_id: 0 for node_id in self.nodes}

        # Calculate in-degrees of nodes
        for node_id, node in self.nodes.items():
            for neighbour in node.out_neighbours:
                in_degrees[neighbour.id] += 1

        # Queue for nodes with no incoming edges
        queue = deque([node_id for node_id, in_degree in in_degrees.items() if in_degree == 0])

        # Topologically sorted nodes
        sorted_nodes = []

        # Perform topological sorting
        while queue:
            node_id = queue.popleft()
            sorted_nodes.append(node_id)

            # Reduce in-degree of neighbors
            for neighbour in self.nodes[node_id].out_neighbours:
                in_degrees[neighbour.id] -= 1

                # Add neighbor to queue if its in-degree becomes 0
                if in_degrees[neighbour.id] == 0:
                    queue.append(neighbour.id)

        return sorted_nodes

    def assign_layers(self):
        # Perform topological sort to get node order
        node_order = self.topological_sort()

        # Dictionary to store layer assignments
        layer_assignments = {node_id: 0 for node_id in self.nodes}

        # Assign layers based on topological order
        for node_id in node_order:
            node = self.nodes[node_id]
            if node.in_neighbours:
                max_predecessor_layer = max(layer_assignments[predecessor.id] for predecessor in node.in_neighbours)
            else:
                max_predecessor_layer = 0
            layer_assignments[node_id] = max_predecessor_layer + 1

        # Update layer attribute of nodes
        for node_id, layer in layer_assignments.items():
            self.nodes[node_id].layer = layer

        return layer_assignments

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
