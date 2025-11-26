import numpy as np

class MoveHeatmap:
    """Tracks move frequency on a board."""

    def __init__(self, ax, board_size=9):
        self.ax = ax
        self.board_size = board_size
        self.move_counts = np.zeros((board_size, board_size), dtype=np.int32)
        self.im = ax.imshow(self.move_counts, cmap='hot', vmin=0, vmax=1)
        ax.set_title("Move Heatmap")
        ax.set_xticks(np.arange(board_size))
        ax.set_yticks(np.arange(board_size))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticklabels([str(i) for i in range(board_size)])
        ax.set_yticklabels([str(i) for i in range(board_size)])
        self.total_moves = 0

    def update(self, moves):
        if not moves:
            return

        updated = False
        for move in moves:
            if move is None:
                continue
            row, col = move
            if row is None or col is None:
                continue
            if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                continue
            self.move_counts[row, col] += 1
            self.total_moves += 1
            updated = True

        if not updated:
            return

        vmax = max(self.move_counts.max(), 1)
        self.im.set_data(self.move_counts)
        self.im.set_clim(vmin=0, vmax=vmax)
        self.ax.set_title(f"Move Heatmap (Total moves: {self.total_moves})")
