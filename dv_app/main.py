import streamlit as st
from data_structures.graph import Graph
from data_structures.node import TreeNode
from helper_functions.step4 import *

colours = {"rome.dot":"#98c412", "LesMiserables.dot": "#6096e6", "JazzNetwork.dot": "#e3cc49",
               "noname.dot":"#bfbfbb","devonshiredebate_withclusters.dot":"#b04dd1"}

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Data Visualization")
    # with st.expander("Note: Complexity and Graph Size"):
    #     st.write("Each step has an associated computational complexity and operates on a graph of size N.")
    #     st.write("Complexity of each step:")
    #     st.write("- Step 1: O(N) for reading and drawing the graph.")
    #     st.write("- Step 2: O(N) for extracting and visualizing trees.")
    #     st.write("- Step 3: O(N^2) for computing a force-directed layout.")
    #     st.write("- Step 4: O(N^2) for some hypothetical operation.")
    #     st.write("- Step 5: O(N) for visualizing multilayer/clustered graphs and performing edge bundling.")
    #     st.write("- Step 6: O(N^2) for performing projections for graphs.")
    #     st.write("Note: N represents the size of the graph.")
    # st.text("Main page. We can add some information here, maybe the link to our paper.")
    pages = {"Step 1": step_1,"Step 2": step_2, "Step 6":step_6, "Step 4": step_4,"Step 5": step_5, "Step 3":step_3}

    with st.sidebar:
        page_step = st.radio(
            "Choose step:",
            (f"Step {i}" for i in range(1,7))
        )
    if page_step:
        pages[page_step]()

def step_1():

    st.title("Step 1: Read and draw a graph")

    example_graphs = {"Rome (N=100)": "rome.dot","Les Misérables network (N=77)": "LesMiserables.dot", "Jazz network (N=198)":"JazzNetwork.dot"}
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())

    g = Graph("datasets/" + example_graphs[selected_graph],colour=colours[example_graphs[selected_graph]])
    bundled= st.toggle("Edge Bundling")
    selected_layout = st.radio("Layout type", ["Random", "Circular"])
    labels = st.toggle("Label Nodes")
    if st.button("Visualize Graph"):
        with st.spinner("Loading..."):
            if selected_layout == "Random":
                g.random_layout()
            if selected_layout == "Circular":
                g.circular_layout()
            g.return_fig(bundled=bundled,labels=labels)

            st.pyplot(g.fig)

def step_2():
    st.title("Step 2: Extract and visualize trees")
    example_graphs = {"Les Misérables network (N=77)": "LesMiserables.dot", "Jazz network (N=198)":"JazzNetwork.dot"}
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())

    g = Graph("datasets/" + example_graphs[selected_graph],colour=colours[example_graphs[selected_graph]])
    selected_graph_traversal = st.radio("Graph traversal type", ["BFS", "DFS"])
    toggle = st.toggle("Non-tree Edges")
    labels = st.toggle("Label Tree Nodes")


    if st.button("Visualize Graph"):
        with st.spinner("Loading..."):
            root = max([(label, node.degree()) for label, node in g.nodes.items()],key=lambda x: x[1])[0] #list(g.nodes.keys())[0]
            if selected_graph_traversal == "DFS":
                g.dfs(root)
                tree_dict = g.dfs_tree
                non_tree_edges = g.dfs_non_tree_edges
            if selected_graph_traversal == "BFS":
                g.bfs(root)
                tree_dict = g.bfs_tree
                non_tree_edges = g.bfs_non_tree_edges

            t = TreeNode(next(iter(tree_dict))).build_tree_from_dict(tree_dict)
            plots = {False : t.draw_tree(labels=labels,colour=colours[example_graphs[selected_graph]]), True: t.draw_tree(labels=labels,non_tree_edges=non_tree_edges,colour=colours[example_graphs[selected_graph]])}
            st.pyplot(plots[toggle])

def step_3():
    st.title("Step 3: Compute a force-directed layout")

    example_graphs = {
        "Les Misérables network (N=77)": "LesMiserables.dot",
        "Rome (N=100)": "rome.dot",
        "Small directed Network (N=24)": "noname.dot",
        "Jazz network (N=198)": "JazzNetwork.dot"
    }
    selected_graph = st.selectbox("Choose an example graph to display", list(example_graphs.keys()))

    # Assuming the Graph class and colours dictionary are defined elsewhere
    g = Graph("datasets/" + example_graphs[selected_graph], colour=colours[example_graphs[selected_graph]])


    # Use columns to place parameters inputs on the right
    col1, col2 = st.columns([3, 1])

    with col1:
        embedder_names = ["Eades", "Fruchterman & Reingold"]
        chosen_embedder = st.radio("Embedder type", embedder_names)
        labels = st.toggle("Label Nodes")

    with col2:


        # Display parameters based on the chosen embedder
        if chosen_embedder == "Fruchterman & Reingold":
            g_const = st.number_input("Gravitational force", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            mag_constant = st.number_input("Magnetic Force", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        elif chosen_embedder == "Eades":
            k_rep = st.number_input("Repulsion constant (k_rep)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            k_spring = st.number_input("Spring constant (k_spring)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)



    if st.button("Visualize Graph"):
        with st.spinner("Loading..."):
            g.random_layout()
            spring_embedders = {
                "Eades": lambda: g.spring_embedder(k_rep=k_rep, k_spring=k_spring),
                "Fruchterman & Reingold": lambda: g.spring_embedder_f(gravitational_constant=g_const, magnetic_constant=mag_constant)
            }

            # Execute the selected embedder function with the specified parameters
            spring_embedders[chosen_embedder]()

            fig = g.return_fig(labels=labels)
            st.pyplot(fig)

def step_4():
    import networkx as nx
    st.title("Step 4: Compute a layered layout")

    example_graphs = {"Small directed Network (N=24)":"noname.dot"}
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())

    g = nx.nx_agraph.read_dot("datasets/" + example_graphs[selected_graph])
    steps = ['Resolve Cycles', 'Layer Assignment', 'Crossing Minimization']
    chosen_step = st.radio("Sugiyama Framework Steps", steps)


    if st.button("Visualize Graph"):
        with st.spinner("Loading..."):
            if chosen_step == "Resolve Cycles":
                plot = plot_trivial_heuristic(g)
                st.pyplot(plot)
            if chosen_step == "Layer Assignment":
                plot = draw_layer_assignment(g)
                st.pyplot(plot)
            if chosen_step == "Crossing Minimization":
                plot = draw_crossing_reduction(g)
                st.pyplot(plot)






def step_5():
    st.title("Step 5: Multilayer/clustered graphs and edge bundling")

    example_graphs = {"Devonshire Debate (N=335)":"devonshiredebate_withclusters.dot"}
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())
    subgraphs = ["Youngest Devonian Strata", "Gap in the Sequence of Devonshi","Dating of the Culm Limestone","Rocks, Fossils and Time","Fossils in Pre-Old Red Sandston",
                        "Other Regions Than Devonshire","Evidence","Universalities","Dating of the Non-Culm"]
    with st.expander("Select Subgraphs"):
        selected_subgraphs = {sg:st.checkbox(sg) for sg in subgraphs}
    s_subgraphs = [sg for sg,b in selected_subgraphs.items() if b]
    edge_bundling = st.toggle("Bundle Intra-layer Edges"), st.toggle("Bundle Subgraph Edges (Not recommended, may take several minutes to compute)")
    labels = st.toggle("Label Nodes")
    if st.button("Visualize Graph"):
        with st.spinner("Loading..."):
            g = Graph("datasets/" + example_graphs[selected_graph],subgraphs=True,selected_subgraphs=s_subgraphs,colour=colours[example_graphs[selected_graph]])
            g.random_layout(subgraphs=True)
            g.spring_embedder_f(ideal_length=.2, gravitational_constant=.1,magnetic_constant=.1)
            g.return_subplots(bundled=edge_bundling,labels=labels)

            st.pyplot(g.fig)

def step_6():
    example_graphs = {
        "Rome (N=100)": "rome.dot",
        "Les Misérables network (N=77)": "LesMiserables.dot",
        "Jazz network (N=198)": "JazzNetwork.dot"
    }
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())

    # Assuming the Graph class and colours dictionary are defined elsewhere
    g = Graph("datasets/" + example_graphs[selected_graph], colour=colours[example_graphs[selected_graph]])

    # Use columns to organize the layout: the radio button and parameters input
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_projection = st.radio("Graph traversal type", ["MDS", "t-SNE", "ISOMAP"])
    with col2:
        # Initialize parameters with None to use them outside of the "if"
        random_state = None
        n_neighbors = None

        # Display parameters input next to the selected method if one is selected
        if selected_projection in ["t-SNE", "MDS"]:
            random_state = st.number_input("Random State", min_value=0, value=0, step=1, format="%d")
            if selected_projection == "t-SNE":
                perplexity = st.number_input("Perplexity", min_value=1, value=7, step=1, format="%d")

        if selected_projection == "ISOMAP":
            # Assuming n_neighbors is relevant for t-SNE and ISOMAP
            n_neighbours = st.number_input("N Neighbours", min_value=1, value=5, step=1, format="%d")

    draw_edges = st.toggle("Draw Edges")
    labels = st.toggle("Label Nodes")

    if st.button("Visualize Graph"):
        g.distances_matrix()
        if selected_projection == "MDS":
            g.mds_coordinates(random_state=random_state)  # Adjust your method calls accordingly
            st.pyplot(g.return_fig(draw_edges=draw_edges,labels=labels))

        elif selected_projection == "t-SNE":
            g.tsne_coordinates(random_state=random_state, perplexity=perplexity)  # Adjust your method calls accordingly
            st.pyplot(g.return_fig(draw_edges=draw_edges,labels=labels))

        elif selected_projection == "ISOMAP":
            g.isomap_coordinates(n_neighbours=n_neighbours)  # Adjust your method calls accordingly
            st.pyplot(g.return_fig(draw_edges=draw_edges,labels=labels))




if __name__ == "__main__":

    main()
