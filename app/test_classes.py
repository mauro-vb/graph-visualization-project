from classes import Node, Graph, Tree
import pygraphviz

dot_file_path = 'Datasets/LesMiserables.dot'

# read dot file
G = pygraphviz.AGraph()
G.read(dot_file_path)

# instantiate own graph
graph = Tree()

for node in G.nodes():
    new_node = Node(label=node.get_name())
    for potential_neighbour in G.nodes():
        if G.has_edge(node, potential_neighbour)*1:
            new_node.add_neighbour(potential_neighbour.get_name())
    graph.add_node(new_node=new_node)

graph.compute_dfs_tree('1')
graph.root.print_tree()
