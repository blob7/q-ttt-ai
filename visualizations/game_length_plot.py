import matplotlib.pyplot as plt

class GameLengthPlot:
    """Histogram of turns per game."""

    def __init__(self, ax):
        self.ax = ax
        self.turns_list = []
        ax.set_xlabel("Turns per Game")
        ax.set_ylabel("Count")

    def update(self, num_turns):
        self.turns_list.append(num_turns)
        self.ax.clear()
        self.ax.hist(self.turns_list, bins=range(1, max(self.turns_list)+2))
        self.ax.set_xlabel("Turns per Game")
        self.ax.set_ylabel("Count")
