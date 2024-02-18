
# Graph Visualization CLI App

This CLI app allows you to visualize a graph using the `plot_graph` function provided in the `plot.py` module.

## Usage

1. First, ensure you have Python installed on your system.

2. Clone or download the repository to your local machine.

3. Navigate to the directory containing the `main.py` file in your terminal.

4. Run the following command:

python main.py <path_to_graph_file> [--xlim XMIN XMAX] [--ylim YMIN YMAX] [--layout {circular, random}] [--axis] [--color COLOR] [--node_tag]

lua


Replace `<path_to_graph_file>` with the path to the file containing the graph data in DOT format.

### Optional Arguments:

- `--xlim XMIN XMAX`: Set custom x-axis limits for the plot. Default is [0, 1].
- `--ylim YMIN YMAX`: Set custom y-axis limits for the plot. Default is [0, 1].
- `--layout {circular, random}`: Choose the layout type for graph visualization. Default is circular.
- `--axis`: Display axis on the plot.
- `--color COLOR`: Choose the color for nodes. Default is green. You can choose from a variety of colors provided by Matplotlib.
- `--node_tag`: Display node numbers on the plot.

### Example:

python main.py graph_data.dot --xlim 0 10 --ylim 0 10 --layout random --axis --color red --node_tag

vbnet


This command visualizes the graph stored in `graph_data.dot` file using a random layout, with custom x-axis and y-axis limits set to [0, 10], displaying axis on the plot, using red color for nodes, and displaying node numbers on the plot.

## Dependencies

- numpy
- matplotlib
- pygraphviz
