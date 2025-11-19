import matplotlib.pyplot as plt
import numpy as np

class MoveHeatmap:
    """Tracks move frequency on a board."""

    def __init__(self, ax, board_size=9):
        self.ax = ax
        self.board_size = board_size
        self.move_counts = np.zeros((board_size, board_size))
        self.im = ax.imshow(self.move_counts, cmap='hot', vmin=0)
        ax.set_title("Move Heatmap")

    def update(self, move):
        row, col = move
        self.move_counts[row, col] += 1
        self.im.set_data(self.move_counts)
        self.im.set_clim(vmin=0, vmax=self.move_counts.max())
