from matplotlib.patches import ConnectionPatch,Circle


class Edge:
    def __init__(self, n1, n2, weight=1, directed=False):
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
        if self.directed:
            if self.inverted:
                self.line = ConnectionPatch(self.circle1.center, self.circle2.center, coordsA=coordsA, coordsB=coordsB, axesA=ax1, axesB=ax2, lw=0.4, color=color, arrowstyle="->")
            else:
                self.line = ConnectionPatch(self.circle1.center, self.circle2.center, coordsA=coordsA, coordsB=coordsB, axesA=ax1, axesB=ax2, lw=0.4, color=color, arrowstyle="->")
        else:
            self.line = ConnectionPatch(self.circle1.center, self.circle2.center, coordsA=coordsA, coordsB=coordsB, axesA=ax1, axesB=ax2, lw=0.4, color=color)

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
