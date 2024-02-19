import argparse
import numpy as np
from plot import plot_graph
from dot_parser import get_adj_matrix
import time
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize a graph using the plot_graph function.")
    parser.add_argument("graph_file", type=str, help="Path to the file containing thegraph")
    parser.add_argument("--xlim", nargs=2, type=float, default=[0, 1], help="Custom x-axis limits for the plot")
    parser.add_argument("--ylim", nargs=2, type=float, default=[0, 1], help="Custom y-axis limits for the plot")
    parser.add_argument("--layout", choices=['circular', 'random'], default='circular', help="Layout type for graph visualization")
    parser.add_argument("--axis", action="store_true", help="Display axis on the plot")
    parser.add_argument("--color", choices=plt.colormaps(), type=str, default="green", help="Node color")
    parser.add_argument("--node_tag", action="store_true", help="Display node numbers on the plot")
    args = parser.parse_args()

    adj_matrix = get_adj_matrix(args.graph_file)

    # Set seed for random layout
    if args.layout == 'random':
        seed = int(time.time())
        print(f"Using seed value: {seed}")
        np.random.seed(seed)

    # Call plot_graph function with provided arguments
    plot_graph(adj_matrix, custom_xlim=args.xlim, custom_ylim=args.ylim, layout_type=args.layout, axis=args.axis, color=args.color, node_tag=args.node_tag)

if __name__ == "__main__":
    main()
