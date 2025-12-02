import numpy as np

class MoveHeatmap:
    """Tracks move frequency on a board."""

    def __init__(self, ax, board_size=9):
        self.ax = ax
        self.board_size = board_size
        self.move_counts = np.zeros((board_size, board_size), dtype=np.int32)
        self.im = ax.imshow(self.move_counts, cmap='hot', vmin=0, vmax=1)
        self.cbar = ax.figure.colorbar(self.im, ax=self.ax)
        self.cbar.set_label("Move frequency")
        ax.set_title("Complete Match Move Landscape")
        ax.set_xticks(np.arange(board_size))
        ax.set_yticks(np.arange(board_size))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticklabels([str(i) for i in range(board_size)])
        ax.set_yticklabels([str(i) for i in range(board_size)])
        self.total_moves = 0
        self.caption = None
        self.ax.figure.subplots_adjust(bottom=0.2)
        self._update_caption()

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
        self._update_caption()

    def _update_caption(self):
        text = f"Total moves: {self.total_moves}"
        if self.caption is None:
            self.caption = self.ax.text(
                0.5,
                -0.2,
                text,
                ha="center",
                va="top",
                transform=self.ax.transAxes,
            )
        else:
            self.caption.set_text(text)
