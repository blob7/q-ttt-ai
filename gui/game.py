import tkinter as tk
from gui.drawer import BoardDrawer
from gui.controller import GameController
from gui.enums import GameMode
from game.environment import GameEnv


class TicTacToeGUI:
    def __init__(self, mode: GameMode, bot1=None, bot2=None):
        self.env = GameEnv()
        self.mode = mode
        self.bot1 = bot1
        self.bot2 = bot2

        # --- UI Setup ---
        self.root = tk.Tk()
        self.root.title("9x9 Directional Tic Tac Toe")

        self.canvas = tk.Canvas(self.root, width=450, height=450)
        self.canvas.grid(row=0, column=0, columnspan=4)

        self.drawer = BoardDrawer(self.canvas, self.env)
        self.controller = GameController(self.root, self.env, self.drawer, mode, bot1, bot2)

        self.drawer.draw_board()
        self.controller.build_ui(self.canvas)

    def run(self):
        self.root.mainloop()
