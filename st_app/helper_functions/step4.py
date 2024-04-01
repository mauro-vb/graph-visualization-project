import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, FancyArrowPatch

def feedback_arc_set(G):
    G_copy = G.copy()
    fas = set()

    while True:
        try:
            cycle = nx.find_cycle(G_copy, orientation='original') # Find a cycle
        except nx.NetworkXNoCycle:   # If no more cycles
            break

        # how many cycles
        cycle_edge_counts = {}
        for edge in cycle:
            u, v, _ = edge
            G_copy.remove_edge(u, v)
            cycle_count = sum(1 for _ in nx.simple_cycles(G_copy))
            cycle_edge_counts[(u, v)] = cycle_count
            G_copy.add_edge(u, v)

        edge_to_remove = max(cycle_edge_counts, key=cycle_edge_counts.get)
        G_copy.remove_edge(*edge_to_remove)
        fas.add(edge_to_remove)

    return fas

def feedback_arc_set_ordered(G, node_order):
    backward_edges = [(u, v) for u, v in G.edges() if node_order.index(u) > node_order.index(v)]

    fas = set()
    G_copy = G.copy()

    # Only consider cycles include a backward edge
    cycles_with_backward_edges = [cycle for cycle in nx.simple_cycles(G_copy) if any(edge in backward_edges for edge in zip(cycle, cycle[1:] + cycle[:1]))]

    for cycle in cycles_with_backward_edges:
        for edge in zip(cycle, cycle[1:] + cycle[:1]):
            if edge in backward_edges:
                fas.add(edge)
                G_copy.remove_edge(*edge)
                break

    assert nx.is_directed_acyclic_graph(G_copy), "The resulting graph is not acyclic."

    return fas

def rev_feedback_arc_set(G, fas):
    rev_fas= set()
    for edge in fas:
        u,v = edge
        rev_fas.add((v,u))
    return rev_fas

def original_graph_without_cycles(G, fas):
    G1 = G.copy()
    for edge in feedback_arc_set(G):
        u,v = edge
        G1.remove_edge(u,v)
    return G1

def dag_graph(G, fas):
    G_a = G.copy()
    G1 =original_graph_without_cycles(G_a, fas)
    for edge in rev_feedback_arc_set(G, fas):
        u,v = edge
        G1.add_edge(u,v)
    return G1

def draw_curved_arrow(ax, A, B, color='black', rad=0.2):
    # choose a "radius" for the arc
    control_point = [(A[0]+B[0])/2, (A[1]+B[1])/2 + rad]
    arrow = FancyArrowPatch(A, B, connectionstyle=f"arc3,rad={rad}",
                            color=color, arrowstyle='-|>',
                            mutation_scale=10.0,
                            lw=1, alpha=0.5)
    ax.add_patch(arrow)


def plot_trivial_heuristic(G):
    fas = feedback_arc_set_ordered(G=G,node_order=[node for node in G.nodes()])
    original_edges = list()
    for edge in original_graph_without_cycles(G, fas).edges():
        u,v =edge
        original_edges.append((u,v))
    reversed_edges = list(rev_feedback_arc_set(G, fas))
    pos = {node: (int(node), 0) for node in G.nodes()}  # Positions for all nodes in a line

    fig, ax = plt.subplots(figsize=(8, 9))

    for node in G.nodes():
        circle = Circle(pos[node], 0.2, fill=True, color='lightblue', zorder=2)
        ax.add_patch(circle)
        plt.text(pos[node][0], pos[node][1], str(node), fontsize=10, ha='center', va='center', zorder=3)

    for edge in original_edges:
        draw_curved_arrow(ax, pos[edge[0]], pos[edge[1]], rad=0.1 + 0.05 * float(edge[0]))


    for edge in reversed_edges:
        draw_curved_arrow(ax, pos[edge[0]], pos[edge[1]], rad=0.1 + 0.1 * float(edge[0]), color='red')
    plt.title('Trivial Heuristic')
    plt.axis('equal')
    plt.axis('off')
    plt.ylim(0,1)
    plt.xlim(-1, len(G.nodes()))  # Set limits to include all nodes and edges

    return fig


def N_outgoing(v, G):
    return {(v, u) for u in G.successors(v)}

def N_incoming(v, G):
    return {(u, v) for u in G.predecessors(v)}

def N(v, G):
    return N_outgoing(v, G).union(N_incoming(v, G))

def heuristic_with_guarantees(G):
    A_prime = set()

    G_copy = G.copy()

    while len(G_copy.nodes) > 0:
        sinks = [v for v in G_copy.nodes if G_copy.in_degree(v) > 0 and G_copy.out_degree(v) == 0]
        for v in sinks:
            A_prime.update(N_incoming(v, G_copy))
            G_copy.remove_node(v)

        isolated_nodes = list(nx.isolates(G_copy))
        G_copy.remove_nodes_from(isolated_nodes)

        sources = [v for v in G_copy.nodes if G_copy.out_degree(v) > 0 and G_copy.in_degree(v) == 0]
        for v in sources:
            A_prime.update(N_outgoing(v, G_copy))
            G_copy.remove_node(v)

        # If graph is non-empty, select a node with max |N_outgoing| - |N_incoming|
        if len(G_copy.nodes) > 0:
            v = max(G_copy.nodes, key=lambda v: len(N_outgoing(v, G_copy)) - len(N_incoming(v, G_copy)))
            A_prime.update(N_outgoing(v, G_copy))
            # remove the neighbors from the graph as well
            G_copy.remove_nodes_from(N_outgoing(v, G_copy))
            G_copy.remove_node(v)

    return A_prime

def original_graph_without_cycles(G, fas):
    G1 = G.copy()
    for edge in feedback_arc_set(G):
        u,v = edge
        G1.remove_edge(u,v)
    return G1

def layer_assignment(G):
    layers = {}
    i = 0
    G_copy = G.copy()
    removed_edges = set()

    while True:
        S = {v for v, d in G_copy.in_degree() if d == 0}
        if not S:
            remaining_nodes = set(G_copy.nodes())
            if not remaining_nodes:
                break
            # Find node with the highest in-degree
            new_source = max(remaining_nodes, key=lambda x: G_copy.in_degree(x))
            S = {new_source}

        layers[i] = S
        G_copy.remove_nodes_from(S)
        i += 1

    return layers, removed_edges

def draw_layered_graph_n(G, layers, fas, ax):
    pos = {}
    for layer, nodes in layers.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (i, layer)

    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.08, color='lightblue', zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, str(node), fontsize=8, ha='center', va='center', zorder=3)

    for edge in G.edges():
        if edge not in fas:
            draw_curved_arrow(ax, pos[edge[0]], pos[edge[1]], color='black')

    for edge in fas:
        draw_curved_arrow(ax, pos[edge[1]], pos[edge[0]], color='red', rad=-0.2)

    ax.set_xlim(-1, max(len(layer) for layer in layers.values()))
    ax.set_ylim(-1, len(layers))
    ax.axis('off')

def draw_graph_with_layers(G, layers, back_edges, fas):
    pos = {}
    # Assign positions to nodes based on their layer
    for layer, nodes in layers.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (i, -layer)  # Layer as y-coordinate, index as x-coordinate

    # Draw the normal edges in black
    nx.draw_networkx_edges(G, pos, edgelist=set(G.edges()) - back_edges, edge_color='black')

    # Draw the back edges in red
    nx.draw_networkx_edges(G, pos, edgelist=back_edges, edge_color='red')

    # Draw the back edges in red
    nx.draw_networkx_edges(G, pos, edgelist=fas, edge_color='red')

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)

    # Draw the node labels
    nx.draw_networkx_labels(G, pos)

    # Reverse the y-axis to have layer 0 at the top
    plt.gca().invert_yaxis()
    plt.title('Layer Assignement with feedback arcs')
    return plt.gca()

def draw_layer_assignment(G):
    node_order = [node for node in G.nodes()]

    fas = feedback_arc_set_ordered(G, node_order)

    G_dag = original_graph_without_cycles(G, fas)
    # Use the function and draw the graph

    layers, back_edges = layer_assignment(G_dag)

    fas = feedback_arc_set(G)

    draw_graph_with_layers(G_dag, layers, back_edges, fas)

def draw_layered_graph_with_straight_edges(G, layers, final_ordering, fas):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    pos = {}
    for layer, nodes in layers.items():
        for node in nodes:
            pos[node] = (final_ordering[layer][node], layer)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    nx.draw_networkx_edges(G, pos, ax=ax)

    nx.draw_networkx_edges(G, pos, edgelist=fas, edge_color='red')

    ax.set_xlim(-1, max(final_ordering[max(layers.keys())].values()) + 10)
    ax.set_ylim(-max(layers.keys()) - 1, 10)
    ax.axis('off')
    return plt.show()

def draw_layered_graph_with_straight_edges(G, layers, final_ordering, fas):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    pos = {}
    layer_widths = {}

    # First pass: determine the width of each layer based on the number of nodes
    for layer, nodes in layers.items():
        # Assuming a uniform horizontal gap of 1 unit between nodes for simplicity
        # You can adjust this gap as needed
        layer_width = len(nodes) - 1
        layer_widths[layer] = layer_width

    # Calculate the total number of layers to adjust the y-position
    total_layers = max(layers.keys())

    # Second pass: assign positions, centering nodes within each layer
    for layer, nodes in layers.items():
        layer_width = layer_widths[layer]
        start_x = -layer_width / 2  # This centers the nodes in the layer

        for index, node in enumerate(nodes):
            # Adjust the y-position so layers go from bottom to top
            pos[node] = (start_x + index, layer)  # Use positive layer index for y-position

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fas, edge_color='red')

    max_width = max(layer_widths.values()) / 2
    ax.set_xlim(-max_width - 1, max_width + 1)
    ax.set_ylim(-1, total_layers + 1)  # Adjust y-axis limits based on total layers
    ax.axis('off')
    return plt.show()

def calculate_barycenter(v, layer_order, G):
    neighbors = N_incoming(v, G) | N_outgoing(v, G)
    valid_neighbors = {u for u, _ in neighbors if u in layer_order}
    if not valid_neighbors:
        return 0
    barycenter = sum(layer_order[u] for u in valid_neighbors) / len(valid_neighbors)
    return barycenter

def introduce_tiny_gaps(barycenters):
    sorted_nodes = sorted(barycenters.items(), key=lambda x: x[1])
    for i, (v, barycenter) in enumerate(sorted_nodes):
        if i > 0 and barycenter == sorted_nodes[i - 1][1]:
            sorted_nodes[i] = (v, barycenter + 0.001 * i)
    return dict(sorted_nodes)

def barycenter_heuristic(G, layers):
    final_ordering = {i: {} for i in layers.keys()}
    final_ordering[0] = {node: i for i, node in enumerate(layers[0])}

    for layer in range(1, max(layers.keys()) + 1):
        barycenters = {}
        for v in layers[layer]:
            barycenters[v] = calculate_barycenter(v, final_ordering[layer - 1], G)
        barycenters_with_gaps = introduce_tiny_gaps(barycenters)
        sorted_nodes_by_barycenter = sorted(barycenters_with_gaps.items(), key=lambda x: x[1])
        final_ordering[layer] = {node: i for i, (node, _) in enumerate(sorted_nodes_by_barycenter)}

    return final_ordering

def calculate_median(v, layer_order, G):
    neighbors = N_incoming(v, G) | N_outgoing(v, G)
    valid_neighbors = [u for u, _ in neighbors if u in layer_order]
    if not valid_neighbors:
        return 0
    sorted_positions = sorted(layer_order[u] for u in valid_neighbors)
    median_position = sorted_positions[len(sorted_positions) // 2]
    return median_position

def median_heuristic(G, layers):
    final_ordering = {i: {} for i in layers.keys()}
    final_ordering[0] = {node: i for i, node in enumerate(layers[0])}

    for layer in range(1, max(layers.keys()) + 1):
        medians = {}
        for v in layers[layer]:
            medians[v] = calculate_median(v, final_ordering[layer - 1], G)
        medians_with_gaps = introduce_tiny_gaps(medians)
        sorted_nodes_by_median = sorted(medians_with_gaps.items(), key=lambda x: x[1])
        final_ordering[layer] = {node: i for i, (node, _) in enumerate(sorted_nodes_by_median)}

    return final_ordering

def iterate_layers_to_minimize_crossings(G, layers, num_iterations=10):
    final_ordering = barycenter_heuristic(G, layers)

    for iteration in range(num_iterations):
        for direction in ('up', 'down'):
            if direction == 'down':
                layer_range = range(1, max(layers.keys()) + 1)
            else:
                layer_range = range(max(layers.keys()) - 1, 0, -1)

            for layer in layer_range:
                adjacent_layer = layer - 1 if direction == 'down' else layer + 1
                barycenters = {
                    v: calculate_barycenter(v, final_ordering[adjacent_layer], G)
                    for v in layers[layer]
                }
                barycenters_with_gaps = introduce_tiny_gaps(barycenters)
                sorted_nodes_by_barycenter = sorted(
                    barycenters_with_gaps.items(), key=lambda x: x[1]
                )
                final_ordering[layer] = {
                    node: i for i, (node, _) in enumerate(sorted_nodes_by_barycenter)
                }

    return final_ordering

def combine_heuristics(G, layers):
    final_ordering = {i: {} for i in layers.keys()}
    final_ordering[0] = {node: i for i, node in enumerate(layers[0])}

    for layer in range(1, max(layers.keys()) + 1):
        barycenters = {}
        medians = {}
        for v in layers[layer]:
            barycenters[v] = calculate_barycenter(v, final_ordering[layer - 1], G)
            medians[v] = calculate_median(v, final_ordering[layer - 1], G)

        # Combine barycenters and medians
        combined_values = {}
        for node in layers[layer]:
            combined_values[node] = (barycenters[node], medians[node])

        combined_values_with_gaps = introduce_tiny_gaps(combined_values)
        sorted_nodes_by_combined = sorted(combined_values_with_gaps.items(), key=lambda x: x[1])
        final_ordering[layer] = {node: i for i, (node, _) in enumerate(sorted_nodes_by_combined)}

    return final_ordering

def draw_crossing_reduction(G):
    node_order = [node for node in G.nodes()]

    fas = feedback_arc_set_ordered(G, node_order)

    G_dag = original_graph_without_cycles(G, fas)
    # Use the function and draw the graph

    layers, back_edges = layer_assignment(G_dag)
    layers = iterate_layers_to_minimize_crossings(G, layers, num_iterations=20)

    fas = feedback_arc_set(G)
    final_ordering =minimize_crossings_with_combined_heuristics(G,layers)

    return draw_layered_graph_with_straight_edges(G, layers, final_ordering, fas)

def reorder_layer_by_combined_heuristic(G, layer, previous_layer_order):
    """Reorder nodes in a layer based on a combination of barycenter and median heuristics."""
    combined_values = {}
    for v in layer:
        barycenter = calculate_barycenter(v, previous_layer_order, G)
        median = calculate_median(v, previous_layer_order, G)
        # Combine barycenter and median by averaging them
        combined_values[v] = (barycenter + median) / 2
    combined_values_with_gaps = introduce_tiny_gaps(combined_values)
    sorted_nodes_by_combined = sorted(combined_values_with_gaps.items(), key=lambda x: x[1])
    return {node: i for i, (node, _) in enumerate(sorted_nodes_by_combined)}

def minimize_crossings_with_combined_heuristics(G, layers, num_iterations=10):
    """Minimize crossings in a layered graph using a combined approach of barycenter and median heuristics."""
    # Initial ordering for the first layer
    final_ordering = {0: {node: i for i, node in enumerate(layers[0])}}

    # Apply combined heuristic iteratively to minimize crossings
    for _ in range(num_iterations):
        for direction in ('down', 'up'):
            layer_range = range(1, max(layers.keys()) + 1) if direction == 'down' else range(max(layers.keys()) - 1, -1, -1)
            for layer in layer_range:
                previous_layer = layer - 1 if direction == 'down' else layer + 1
                final_ordering[layer] = reorder_layer_by_combined_heuristic(G, layers[layer], final_ordering[previous_layer])

    return final_ordering
