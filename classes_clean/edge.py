from matplotlib.patches import ConnectionPatch

class Edge:
    def __init__(self, n1, n2, weight=1, directed=False):
        self.directed = directed
        self.node1, self.node2 = n1, n2
        self.circle1 = n1.circle
        self.circle2 = n2.circle
        self.inverted = False
        self.weight = weight
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
