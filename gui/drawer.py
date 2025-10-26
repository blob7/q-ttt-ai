from enum import Enum

CELL_SIZE = 50
GRID_COLOR = "black"


class BoardColor(Enum):
    X_TEXT = "blue"
    X_WIN_BACKGROUND = "blue"

    O_TEXT = "red"
    O_WIN_BACKGROUND = "red"

    VALID_MOVE_BACKGROUND = "#fffdf0"



class BoardDrawer:
    def __init__(self, canvas, env):
        self.canvas = canvas
        self.env = env

    def draw_board(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_symbols()

        winner, winning_cells = self.env.game.check_winner(return_cells=True)

        if winner and winning_cells:
            self._highlight_winner(winner, winning_cells)
        elif not winner:
            for r, c in self.env.get_valid_moves():
                self._highlight_cell(r, c)

    def _draw_grid(self):
        for i in range(10):
            self.canvas.create_line(0, i * CELL_SIZE, 9 * CELL_SIZE, i * CELL_SIZE, fill=GRID_COLOR)
            self.canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, 9 * CELL_SIZE, fill=GRID_COLOR)

    def _draw_symbols(self):
        board = self.env.game.board
        for r in range(9):
            for c in range(9):
                self._draw_symbol(r, c, board[r, c])

    def _draw_symbol(self, r, c, val):
        if val == 0:
            return
        text = "X" if val == 1 else "O"
        color = BoardColor.X_TEXT.value if val == 1 else BoardColor.O_TEXT.value
        x, y = c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2
        self.canvas.create_text(x, y, text=text, fill=color, font=("Arial", 24))

    def _highlight_cell(self, r, c):
        self.canvas.create_rectangle(
            c * CELL_SIZE, r * CELL_SIZE,
            c * CELL_SIZE + CELL_SIZE, r * CELL_SIZE + CELL_SIZE,
            fill=BoardColor.VALID_MOVE_BACKGROUND.value, width=3
        )

    def _highlight_winner(self, winner, winning_cells):
        color = BoardColor.X_WIN_BACKGROUND.value if winner == 1 else BoardColor.O_WIN_BACKGROUND.value
        for r, c in winning_cells:
            self.canvas.create_rectangle(
                c * CELL_SIZE, r * CELL_SIZE,
                c * CELL_SIZE + CELL_SIZE, r * CELL_SIZE + CELL_SIZE,
                fill=color, stipple="gray25", width=0
            )
