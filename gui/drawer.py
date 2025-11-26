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
        self.cells: int = cells
        self.base_cell_size: int = cell_size
        self._cell_size: int = cell_size
        self._offset_x: int = 2
        self._offset_y: int = 2

        # Create internal canvas and pack it so the view can grid this Frame
        # Add a small internal inset so lines and borders are not clipped at
        # the canvas edges. `offset` is applied to all drawing coordinates.
        self.canvas = tk.Canvas(
            self,
            width=self.cells * self.base_cell_size + 4,
            height=self.cells * self.base_cell_size + 4,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        # Redraw board whenever the canvas size changes so the board scales.
        self.draw_board()

    def _compute_geometry(self) -> None:
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)

        if canvas_w <= 1 or canvas_h <= 1:
            self._cell_size = self.base_cell_size
            self._offset_x = 2
            self._offset_y = 2
            return

        usable_w = canvas_w - 4
        usable_h = canvas_h - 4
        cell_size = int(max(min(usable_w / self.cells, usable_h / self.cells), 10))
        board_px = cell_size * self.cells

        offset_x = max(2, (canvas_w - board_px) // 2)
        offset_y = max(2, (canvas_h - board_px) // 2)

        self._cell_size = cell_size
        self._offset_x = int(offset_x)
        self._offset_y = int(offset_y)

    def draw_board(self):
        self._compute_geometry()
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
        size_px = self.cells * self._cell_size
        start_x = self._offset_x
        start_y = self._offset_y
        end_x = start_x + size_px
        end_y = start_y + size_px

        for i in range(self.cells + 1):
            y = start_y + i * self._cell_size
            x = start_x + i * self._cell_size
            self.canvas.create_line(start_x, y, end_x, y, fill=GRID_COLOR)
            self.canvas.create_line(x, start_y, x, end_y, fill=GRID_COLOR)

        self.canvas.create_rectangle(
            start_x,
            start_y,
            end_x,
            end_y,
            outline=GRID_COLOR,
            width=2,
        )

    def _draw_symbols(self):
        board = self.env.get_board()
        for r in range(self.cells):
            for c in range(self.cells):
                self._draw_symbol(r, c, board[r, c])

    def map_pixel_to_cell(self, x: int, y: int):
        """Convert canvas pixel coordinates to (row, col). Returns None if
        the coordinates are outside the board bounds.
        """
        if x < self._offset_x or y < self._offset_y:
            return None
        col = (int(x) - self._offset_x) // self._cell_size
        row = (int(y) - self._offset_y) // self._cell_size
        if row < 0 or row >= self.cells or col < 0 or col >= self.cells:
            return None
        return row, col

    def _draw_symbol(self, r, c, val):
        if val == 0:
            return
        text = "X" if val == 1 else "O"
        color = BoardColor.X_TEXT.value if val == 1 else BoardColor.O_TEXT.value
        x = self._offset_x + c * self._cell_size + self._cell_size / 2
        y = self._offset_y + r * self._cell_size + self._cell_size / 2
        font_size = max(int(self._cell_size * 0.5), 14)
        self.canvas.create_text(x, y, text=text, fill=color, font=("Arial", font_size))

    def _highlight_cell(self, r, c):
        x0 = self._offset_x + c * self._cell_size
        y0 = self._offset_y + r * self._cell_size
        x1 = x0 + self._cell_size
        y1 = y0 + self._cell_size
        self.canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=BoardColor.VALID_MOVE_BACKGROUND.value, width=3
        )
        self._draw_cell_coords(r, c)

    def _highlight_winner(self, winner, winning_cells):
        color = BoardColor.X_WIN_BACKGROUND.value if winner == 1 else BoardColor.O_WIN_BACKGROUND.value
        for r, c in winning_cells:
            x0 = self._offset_x + c * self._cell_size
            y0 = self._offset_y + r * self._cell_size
            x1 = x0 + self._cell_size
            y1 = y0 + self._cell_size
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=color, stipple="gray25", width=0
            )

    def _draw_cell_coords(self, r: int, c: int) -> None:
        pad_x = max(int(self._cell_size * 0.08), 4)
        pad_y = max(int(self._cell_size * 0.24), 12)
        x = self._offset_x + c * self._cell_size + pad_x
        y = self._offset_y + r * self._cell_size + pad_y
        font_size = max(int(self._cell_size * 0.18), 8)
        self.canvas.create_text(
            x,
            y,
            text=f"{r},{c}",
            fill="#555",
            font=("Arial", font_size),
            anchor="nw"
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
        return self._cell_size

    @property
    def cell_size(self) -> int:  # Backwards compatibility for existing callers.
        return self._cell_size
