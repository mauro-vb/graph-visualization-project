import streamlit as st
from classes_clean.graph import Graph
import matplotlib.pyplot as plt


def main():
    st.title("Data Visualization")
    st.text("Main page. We can add some information here, maybe the link to our paper.")
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
    example_graphs = {"Les Misérables network": "LesMiserables.dot", "Jazz network":"JazzNetwork.dot"}
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())

    g = Graph("Datasets/" + example_graphs[selected_graph])

    selected_layout = st.radio("Layout type", ["Random", "Circular"])
    if selected_layout == "Random":
        g.random_layout()
    if selected_layout == "Circular":
        g.circular_layout()
    g.return_fig()

    st.pyplot(g.fig)

def step_2():
    st.title("Step 2: Extract and visualize trees")
    example_graphs = {"Les Misérables network": "LesMiserables.dot", "Jazz network":"JazzNetwork.dot"}
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())

    g = Graph("Datasets/" + example_graphs[selected_graph])
    selected_graph_traversal = st.radio("Graph traversal type", ["DFS", "BFS"])
    if selected_graph_traversal == "DFS":
        g.dfs('1')
        st.text(g.dfs_tree)
    if selected_graph_traversal == "BFS":
        g.bfs('1')
        st.text(g.bfs_tree)

def step_3():
    pass

def step_4():
    pass

def step_5():
    pass

def step_6():
    from sklearn.manifold import MDS
    from sklearn.manifold import Isomap
    from sklearn.manifold import TSNE
    st.title("Step 6: Projections for graphs")
    example_graphs = {"Les Misérables network": "LesMiserables.dot", "Jazz network":"JazzNetwork.dot"}
    selected_graph = st.selectbox("Choose an example graph to display", example_graphs.keys())

    g = Graph("Datasets/" + example_graphs[selected_graph])
    selected_projection = st.radio("Graph traversal type", ["MDS", "t-SNE", "ISOMAP"])
    D = g.distances_matrix()
    fig = plt.figure()
    if selected_projection == "MDS":
        # MDS with adjusted parameters
        mds = MDS(n_components=2, dissimilarity="euclidean", random_state=6)
        results = mds.fit(D)
        coords = results.embedding_
        plt.scatter(coords[:, 0], coords[:, 1], marker = 'o')

    if selected_projection == "t-SNE":
        # t-SNE with adjusted parameters
        tsne = TSNE(n_components=2, perplexity=10, learning_rate=1, random_state=0, metric='euclidean')
        Y = tsne.fit_transform(D)
        plt.scatter(Y[:, 0], Y[:, 1], marker = 'o')

    if selected_projection == "ISOMAP":
        # ISOMAP with adjusted parameters
        iso = Isomap(n_neighbors=15, n_components=2)
        iso.fit(D)
        manifold_2Da = iso.transform(D)
        plt.scatter(manifold_2Da[:, 0], manifold_2Da[:, 1], marker = 'o')
    st.pyplot(fig)





if __name__ == "__main__":

    main()
