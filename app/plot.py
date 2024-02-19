import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch

def circular_layout(N, center=(0, 0), radius=1):
    cx, cy = center
    positions = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        positions.append((x, y))
    return positions

def random_layout(N, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
    positions = []
    for i in range(N):
        x = np.random.uniform(x_range[0], x_range[1])  # Generate random x-coordinate
        y = np.random.uniform(y_range[0], y_range[1])  # Generate random y-coordinate
        positions.append((x, y))
    return positions



def plot_graph(adj_matrix, custom_xlim = (0,1), custom_ylim = (0,1), layout_type = 'circular', axis=False, color = 'green', node_tag = True):

    # get graph size
    N = len(adj_matrix)

    # get node positions based on desired layout
    if layout_type == 'circular':
        positions = circular_layout(N, center=((custom_xlim[0]+custom_xlim[1]/2), (custom_ylim[0]+custom_ylim[1]/2)), radius=(custom_xlim[0]+custom_xlim[1]/2))
    if layout_type == 'random':
        positions = random_layout(N, x_range=custom_xlim,y_range=custom_ylim)
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()

    # making slightly bigger x_lim and y_lim for better visualization
    x_lim = custom_xlim[0]- (custom_xlim[1]/20) , custom_xlim[1]+ (custom_xlim[1]/20)
    y_lim = custom_ylim[0]- (custom_ylim[1]/20) , custom_ylim[1]+ (custom_ylim[1]/20)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Calculate appropriate node radius based on graph size (N) and bounding box (x_lim,y_lim)
    node_radius = (custom_xlim[1] - custom_xlim[0]) / (5 * np.sqrt(N))
    edge_lw = min((custom_xlim[1] - custom_xlim[0]) / (2 * np.sqrt(N)), 0.2)

    # draw nodes based on positions
    for node,pos in enumerate(positions):
        ax.add_patch(Circle(xy=pos,radius= node_radius, color = color, alpha=.5))
        if node_tag:
            plt.text(*pos, str(node), size=7, ha='center', va='center',alpha=.7)

    # draw edges based on adjacency matrix
    for i in range(N): # loop over rows
        for j in range(N): # loop over columns
            if adj_matrix[i, j] == 1: # check if nodes are adjacent
                start_edge = positions[i] # (some_x, some_y)
                end_edge = positions[j] # (some_x, some_y)

                ax.add_patch(ConnectionPatch(start_edge,end_edge,'data',lw=edge_lw,color='grey'))


    plt.axis(axis)
    plt.show();
