## Graph Visualization App

This repository contains a Python application for visualizing graphs using Matplotlib. You can generate visualizations of your graphs and customize various aspects of the plot.

### Installation

1. Clone or download the repository to your local machine.
2. Navigate to the directory containing the `main.py` file in your terminal.
3. Install the required dependencies using pip: `pip install -r requirements.txt`.

### Usage

#### Running the Streamlit App

1. Navigate to the directory containing the `st_app.py` file in your terminal.
2. Run the command: `streamlit run st_app.py`.
3. Access the app in your web browser at [http://localhost:8501](http://localhost:8501).

#### Running the Graph Visualization Script

1. Navigate to the directory containing the `main.py` file in your terminal.
2. Run the command: `python main.py <path_to_graph_file> [--options]`.

   Replace `<path_to_graph_file>` with the path to the graph file in DOT format.

#### Optional Arguments

- `--options`: Add optional arguments like `--xlim`, `--ylim`, `--layout`, `--axis`, `--color`, and `--node_tag` for customization.

#### Example

```bash
python main.py graph_data.dot --xlim 0 10 --ylim 0 10 --layout random --axis --color red --node_tag
