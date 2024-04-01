# Graph Visualization System

This repository contains a comprehensive system designed for the visualization of relational and high-dimensional data. The project is structured into seven distinct steps, each focusing on different aspects of graph visualization. The goal is to create intuitive and informative visual representations that can facilitate the understanding of complex network structures.

## Project Structure

The project is divided into the following steps:

- **Step 1: Read and Draw a Graph**
- **Step 2: Extract and Visualize Trees**
- **Step 3: Compute a Force Directed Layout**
- **Step 4: Compute a Layered Layout**
- **Step 5: Multilayer/Clustered Graphs and Edge Bundling**
- **Step 6: Projections for Graphs**
- **Step 7: Quality Measurement of Graph Projections**

Each step is implemented in a Jupyter Notebook (`.ipynb` format) which details the process, from reading graph data to applying sophisticated layout algorithms.

## Installation

Clone the repository and navigate to the cloned directory:

git clone -b masoud https://git.science.uu.nl/dataviz/dataviz.git
cd dataviz

Once inside the directory, you can switch to the `masoud` branch if needed:

git checkout masoud

Open the `.ipynb` file for each step to view the implementation.

## Usage

Each notebook is self-contained and includes both the implementation of the visualization step and a detailed explanation of the methods used. You can run each cell sequentially to see the results of the visualizations.

## Step 1: Read and Draw a Graph

In this step, we read graph data from a file and generate a node-link diagram visualization. We use NetworkX to handle the graph structure and Matplotlib for visualization.

### Features
- Load graph from `.dot` file
- Build data structures for graph representation
- Assign positions to nodes using user-defined functions
- Draw the graph with customized layouts
- Computational complexity analysis of the layout algorithm
- Pros and cons discussion of the chosen layout

## Step 2 to Step 7

The following steps build upon the initial graph drawing, introducing more advanced visualization techniques such as tree extraction, force-directed layouts, layered layouts for DAGs, multilayer and clustered graph visualization, graph projections, and quality measurements for these projections.

Each step is documented within its respective notebook with code, commentary, and visualization outputs.

## Contributing

Contributions to improve the visualization system are welcome. Please fork the repository, make your changes, and submit a pull request with a clear description of your improvements.