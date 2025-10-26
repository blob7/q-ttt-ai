import tkinter as tk
from game.environment import GameEnv
from game.utils import print_board
from enum import Enum

CELL_SIZE = 50  # pixels
GRID_COLOR = "black"
X_COLOR = "blue"
O_COLOR = "red"
HIGHLIGHT_COLOR = "lightgreen"

class GameMode(Enum):
    BOT_V_BOT = "bot vs bot"
    PLAYER_V_PLAYER = "player vs player"
    PLAYER_V_BOT = "player vs bot"

class TicTacToeGUI:
    def __init__(self, mode: GameMode, bot1 = None, bot2 = None):
        """
        vs_bot: if True, second player is AI
        bot: a callable that takes (env) and returns a move (row, col)
        """
        self.env = GameEnv()
        self.mode = mode
        if mode == GameMode.PLAYER_V_BOT and bot1 is None:
            raise ValueError("Bot1 must be provided for PLAYER_V_BOT mode")
        self.bot1 = bot1
        self.bot2 = bot2
        self.bot_speed = 500  # ms delay between bot moves

        self.root = tk.Tk()
        self.root.title("9x9 Directional Tic Tac Toe")

        # Canvas for board
        self.canvas = tk.Canvas(self.root, width=CELL_SIZE*9, height=CELL_SIZE*9)
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<Button-1>", self.handle_click)

        # Buttons
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset)
        self.reset_button.grid(row=1, column=0, sticky="ew")

        self.status_label = tk.Label(self.root, text="Player X's turn", font=("Arial", 14))
        self.status_label.grid(row=1, column=1, sticky="ew")

        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.destroy)
        self.quit_button.grid(row=1, column=3, sticky="ew")

        if self.mode != GameMode.PLAYER_V_PLAYER:
            tk.Label(self.root, text="Speed (ms):").grid(row=2, column=0)
            self.speed_slider = tk.Scale(
                self.root,
                from_=100,
                to=2000,
                orient="horizontal",
                command=self.update_speed
            )
            self.speed_slider.set(self.bot_speed)
            self.speed_slider.grid(row=2, column=1, columnspan=2, sticky="ew")

        # Play button (only for bot vs bot)
        if self.mode == GameMode.BOT_V_BOT:
            self.play_button = tk.Button(self.root, text="Play", command=self.start_bot_vs_bot)
            self.play_button.grid(row=1, column=2, sticky="ew")

        self.draw_board()

    def update_speed(self, val):
        self.bot_speed = int(val)


    def draw_board(self):
        """Draw the grid and pieces."""
        self.canvas.delete("all")
        # Draw grid lines
        for i in range(10):
            self.canvas.create_line(0, i*CELL_SIZE, 9*CELL_SIZE, i*CELL_SIZE, fill=GRID_COLOR)
            self.canvas.create_line(i*CELL_SIZE, 0, i*CELL_SIZE, 9*CELL_SIZE, fill=GRID_COLOR)

        # Draw pieces
        board = self.env.game.board
        for r in range(9):
            for c in range(9):
                x0 = c*CELL_SIZE
                y0 = r*CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                if board[r, c] == 1:
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text="X", fill=X_COLOR, font=("Arial", 24))
                elif board[r, c] == -1:
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text="O", fill=O_COLOR, font=("Arial", 24))

        # Highlight valid moves
        if not self.env.game.check_winner():
            for r, c in self.env.get_valid_moves():
                x0 = c*CELL_SIZE
                y0 = r*CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=HIGHLIGHT_COLOR, width=3)

    def handle_click(self, event):
        if self.mode == GameMode.BOT_V_BOT:
            return  # disable clicking

        if self.env.game.check_winner():
            return

        row, col = event.y // CELL_SIZE, event.x // CELL_SIZE
        if (row, col) not in self.env.get_valid_moves():
            return

        self.env.step((row, col))
        self.draw_board()
        self.update_status()

        if self.mode == GameMode.PLAYER_V_BOT and not self.env.game.check_winner():
            if self.bot1 is not None:
                move = self.bot1(self.env)
                self.env.step(move)
                self.draw_board()
                self.update_status()
            else:
                self.status_label.config(text="Error: Bot1 is not defined.")

    def update_status(self):
            winner = self.env.game.check_winner()
            if winner == 1:
                self.status_label.config(text="Player X wins!")
            elif winner == -1:
                self.status_label.config(text="Player O wins!")
            elif winner == 0:
                self.status_label.config(text="Draw!")
            else:
                current = "X" if self.env.game.current_player == 1 else "O"
                self.status_label.config(text=f"Player {current}'s turn")

    def reset(self):
        self.env.reset()
        self.draw_board()
        self.status_label.config(text="Player X's turn")

    def start_bot_vs_bot(self):
        self.play_button.config(state="disabled")
        self.root.after(self.bot_speed, self.run_bot_vs_bot)

    def run_bot_vs_bot(self):
        if self.env.game.check_winner():
            self.update_status()
            return

        current_bot = self.bot1 if self.env.game.current_player == 1 else self.bot2
        if current_bot is None:
            self.status_label.config(text="Error: Bot is not defined for current player.")
            return
        move = current_bot(self.env)
        self.env.step(move)
        self.draw_board()
        self.update_status()

        if not self.env.game.check_winner():
            self.root.after(self.bot_speed, self.run_bot_vs_bot)

    def run(self):
        self.root.mainloop()