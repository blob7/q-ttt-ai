from enum import Enum
from typing import Callable
import tkinter as tk

DEFAULT_CELL_SIZE = 50
DEFAULT_CELLS = 9
GRID_COLOR = "black"


class BoardColor(Enum):
    X_TEXT = "blue"
    X_WIN_BACKGROUND = "blue"

    O_TEXT = "red"
    O_WIN_BACKGROUND = "red"

    VALID_MOVE_BACKGROUND = "#fffdf0"



class BoardDrawer(tk.Frame):
    def __init__(self, parent, env, cells: int = DEFAULT_CELLS, cell_size: int = DEFAULT_CELL_SIZE):
        super().__init__(parent)
        self.env = env
        self.cell_size: int = cell_size
        self.cells: int = cells

        # Create internal canvas and pack it so the view can grid this Frame
        self.canvas = tk.Canvas(self, width=self.cells * self.cell_size, height=self.cells * self.cell_size)
        self.canvas.pack(fill="both", expand=True)

    def draw_board(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_symbols()

        winner, winning_cells = self.env.check_winner(return_cells=True)

        if winner and winning_cells:
            self._highlight_winner(winner, winning_cells)
        elif not winner:
            for r, c in self.env.get_valid_moves():
                self._highlight_cell(r, c)

    def _draw_grid(self):
        for i in range(self.cells + 1):
            self.canvas.create_line(0, i * self.cell_size, self.cells * self.cell_size, i * self.cell_size, fill=GRID_COLOR)
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.cells * self.cell_size, fill=GRID_COLOR)

    def _draw_symbols(self):
        board = self.env.get_board()
        for r in range(self.cells):
            for c in range(self.cells):
                self._draw_symbol(r, c, board[r, c])

    def map_pixel_to_cell(self, x: int, y: int):
        """Convert canvas pixel coordinates to (row, col). Returns None if
        the coordinates are outside the board bounds.
        """
        if x < 0 or y < 0:
            return None
        col = int(x) // self.cell_size
        row = int(y) // self.cell_size
        if row < 0 or row >= self.cells or col < 0 or col >= self.cells:
            return None
        return row, col

    def _draw_symbol(self, r, c, val):
        if val == 0:
            return
        text = "X" if val == 1 else "O"
        color = BoardColor.X_TEXT.value if val == 1 else BoardColor.O_TEXT.value
        x, y = c * self.cell_size + self.cell_size / 2, r * self.cell_size + self.cell_size / 2
        self.canvas.create_text(x, y, text=text, fill=color, font=("Arial", 24))

    def _highlight_cell(self, r, c):
        self.canvas.create_rectangle(
            c * self.cell_size, r * self.cell_size,
            c * self.cell_size + self.cell_size, r * self.cell_size + self.cell_size,
            fill=BoardColor.VALID_MOVE_BACKGROUND.value, width=3
        )

    def _highlight_winner(self, winner, winning_cells):
        color = BoardColor.X_WIN_BACKGROUND.value if winner == 1 else BoardColor.O_WIN_BACKGROUND.value
        for r, c in winning_cells:
            self.canvas.create_rectangle(
                c * self.cell_size, r * self.cell_size,
                c * self.cell_size + self.cell_size, r * self.cell_size + self.cell_size,
                fill=color, stipple="gray25", width=0
            )


    def bind_cell_click(self, callback: Callable[[int, int], None]) -> None:
        """Bind a handler that receives (row, col) when a board cell is
        clicked. This keeps pixel->cell mapping inside the Drawer.
        """
        def _handler(event):
            res = self.map_pixel_to_cell(event.x, event.y)
            if res is None:
                return
            r, c = res
            try:
                callback(r, c)
            except Exception:
                # Do not let UI exceptions bubble out of the drawer
                return

        self.canvas.bind("<Button-1>", _handler)

    def get_cell_size(self) -> int:
        """Return the configured cell size for coordinate calculations."""
        return self.cell_size
