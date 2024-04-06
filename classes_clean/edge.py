from matplotlib.patches import ConnectionPatch,Circle


class Edge:
    def __init__(self, n1, n2, weight=1, directed=False):
        """
        Initializes an Edge instance between two nodes with optional weight and direction.

        Args:
            n1: The starting node of the edge.
            n2: The ending node of the edge.
            weight (int, optional): The weight of the edge. Defaults to 1.
            directed (bool, optional): Indicates if the edge is directed. Defaults to False.
        """
        self.directed = directed
        self.node1, self.node2 = n1, n2
        self.circle1 = n1.circle
        self.circle2 = n2.circle
        self.inverted = False
        self.weight = weight
        self.fig_coordinates = None
        if self.directed:
            if self.inverted:
                self.line = ConnectionPatch(self.circle1.center, self.circle2.center, "data", "data", lw=0.7, color='red', arrowstyle="->")
            else:
                self.line = ConnectionPatch(self.circle1.center, self.circle2.center, "data", "data", lw=0.7, color='red', arrowstyle="->")
        else:
            self.line = ConnectionPatch(self.circle1.center, self.circle2.center, "data", "data", lw=0.7, color='grey',zorder=1)

    def update_line(self, ax1=None, ax2=None, color="grey", coordsA='data', coordsB='data'):
        """
        Updates the line representation of the edge, allowing for changes in axes, color, and coordinates system.

        Args:
            ax1, ax2 (matplotlib.axes.Axes, optional): Axes instances for the start and end nodes. Defaults to None.
            color (str, optional): Color of the edge line. Defaults to "grey".
            coordsA, coordsB (str, optional): Coordinate systems for the start and end points. Defaults to 'data'.
        """
        # Adjust the visual style based on edge properties
        lw = 0.4  # Line width
        arrowstyle = "->" if self.directed else "-"  # Arrow style for directed edges
        self.line = ConnectionPatch(self.circle1.center, self.circle2.center, coordsA=coordsA, coordsB=coordsB,
                                    axesA=ax1, axesB=ax2, lw=lw, color=color, arrowstyle=arrowstyle)


    def get_fig_coordinates(self, ax1, ax2):
        """
        Transforms edge node coordinates from data space to figure space.

        Args:
            ax1 (matplotlib.axes.Axes): Axes instance for node1.
            ax2 (matplotlib.axes.Axes): Axes instance for node2.

        Returns:
            tuple: A tuple containing the start and end coordinates of the edge in figure space.
        """
        fig = ax1.figure

        # Transform from data space to display space
        start_display = ax1.transData.transform(self.node1.circle.center)
        end_display = ax2.transData.transform(self.node2.circle.center)

        # Transform from display space to figure space
        transFigure = fig.transFigure.inverted()
        start_fig = transFigure.transform(start_display)
        end_fig = transFigure.transform(end_display)
        self.fig_coordinates = start_fig, end_fig
        return start_fig, end_fig
