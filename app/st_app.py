import streamlit as st
import matplotlib.pyplot as plt
from classes import Node, Graph, Tree
import pygraphviz
import numpy as np
import os

# Main Streamlit app
def main():
    st.title("Graph and Tree Visualization App")
    datasets = ["Datasets/" + dataset for dataset in os.listdir("Datasets") if dataset.endswith(".dot")]
    selected_file = st.selectbox('select a graph to visualize', datasets)
    if selected_file is not None:
        G = pygraphviz.AGraph(selected_file)
        visualization_options = ["Select", "Tree", "Graph"]
        visualization_type = st.selectbox("Choose Visualization Type", visualization_options)

        if visualization_type == "Tree":

            graph = Tree()
            for node in G.nodes():
                new_node = Node(label=node.get_name())
                for potential_neighbour in G.nodes():
                    if G.has_edge(node, potential_neighbour):
                        new_node.add_neighbour(potential_neighbour.get_name())
                graph.add_node(new_node)

            traversal_options = ["DFS", "BFS"]
            traversal_type = st.selectbox("Choose Tree Traversal Type", traversal_options)
            root_label = st.text_input("Enter the root label for your tree:","1")

            if traversal_type == "DFS":
                graph.compute_dfs_tree(root_label)
                fig = graph.root.draw_tree()
                st.pyplot(fig=fig)
            elif traversal_type == "BFS":
                graph.compute_bfs_tree(root_label)
                fig = graph.root.draw_tree()
                st.pyplot(fig=fig)

        elif visualization_type == "Graph":
            layout_options = ["Random", "Circular"]
            layout_type = st.selectbox("Choose Graph Layout Type", layout_options)
            graph = Graph()

            for node in G.nodes():
                new_node = Node(label=node.get_name())
                for potential_neighbour in G.nodes():
                    if G.has_edge(node, potential_neighbour):
                        new_node.add_neighbour(potential_neighbour.get_name())
                graph.add_node(new_node)

            if layout_type == "Random":
                graph.generate_random_coordinates()
            elif layout_type == "Circular":
                graph.generate_circular_coordinates()
            fig = graph.plot_graph()
            st.pyplot(fig=fig)

if __name__ == "__main__":
    main()
