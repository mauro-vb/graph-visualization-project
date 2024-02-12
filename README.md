# Step 1: Read and Draw a Graph

In this step, we aim to read a graph from a given file and visualize it using Python. We use the NetworkX library to read the graph and matplotlib to visualize it.

## Reading the Graph

We start by reading the graph from a `.dot` file using the `read_dot` function from NetworkX. The graph is stored in the variable `G`. We then extract the edges and nodes from the graph and store them in `edge_list` and `node_list` respectively. Each edge is a tuple of two integers representing the nodes that the edge connects, and each node is an integer.

## Building the Adjacency List

Next, we build an adjacency list from the edge list. The adjacency list is a dictionary where the keys are the nodes and the values are lists of nodes that are connected to the key node by an edge. This gives us a convenient way to look up the neighbors of a node.

## Assigning Positions to Nodes

We then assign a random position to each node. The positions are stored in a dictionary `pos` where the keys are the nodes and the values are 2D numpy arrays representing the positions of the nodes.

## Drawing the Graph

To draw the graph, we create a matplotlib figure and axes. We then iterate over the nodes and draw a circle at each node's position. We also add a text label at each node's position.

Next, we iterate over the edges and draw an arrow from the source node to the target node for each edge. The arrows are drawn using the `ConnectionPatch` class from matplotlib.

Finally, we set the limits of the axes and turn off the axis lines and labels with `plt.axis('off')`. We then display the figure with `plt.show()`.

## Computational Complexity

The computational complexity of the layout algorithm is O(n), where n is the number of nodes. This is because we assign a position to each node independently of the other nodes.

## Pros and Cons of the Layout

The main advantage of this layout is that we have complete control over the positioning of the nodes. However, the layout is not optimized and can result in issues like node overlap or too many edge crossings.

## Bonus: Circular Layout

As a bonus, we implemented a circular layout. In this layout, the nodes are positioned in a circle. The position of each node is determined by its index and the total number of nodes. We then redraw the graph with the new positions.

The circular layout algorithm also has a computational complexity of O(n). The resulting layout has less overlap and fewer edge crossings compared to the random layout.
