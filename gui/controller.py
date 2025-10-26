import tkinter as tk
from gui.enums import GameMode
from gui.components import SpeedInput, PlayButton

class GameController:
    def __init__(self, root, env, drawer, mode, bot1=None, bot2=None):
        self.root = root
        self.env = env
        self.drawer = drawer
        self.mode = mode
        self.bot1 = bot1
        self.bot2 = bot2
        self.bot_speed = 500

    def build_ui(self, canvas):
        canvas.bind("<Button-1>", self.on_click)

        self.status = tk.Label(self.root, text="Player X's turn", font=("Arial", 14))
        self.status.grid(row=1, column=1, sticky="ew")

        tk.Button(self.root, text="Reset", command=self.reset).grid(row=1, column=0, sticky="ew")
        tk.Button(self.root, text="Quit", command=self.root.destroy).grid(row=1, column=3, sticky="ew")

        # --- Speed input (for all bot modes) ---
        if self.mode in (GameMode.PLAYER_V_BOT, GameMode.BOT_V_BOT):
            self.speed_input = SpeedInput(self.root, default_speed=self.bot_speed, on_change=self.update_speed)
            self.speed_input.grid(row=2, column=1, sticky="ew", pady=5)

        # --- Play button (only for Bot vs Bot) ---
        if self.mode == GameMode.BOT_V_BOT:
            self.play_button = PlayButton(self.root, self.start_bot_vs_bot)
            self.play_button.grid(row=1, column=2, sticky="ew")

    def update_speed(self, val):
        self.bot_speed = int(val)

    def on_click(self, event):
        if self.mode == GameMode.BOT_V_BOT or self.env.check_winner():
            return

        row, col = event.y // 50, event.x // 50
        if (row, col) not in self.env.get_valid_moves():
            return

        self.env.step((row, col))
        self.drawer.draw_board()
        self.update_status()

        if self.mode == GameMode.PLAYER_V_BOT and not self.env.check_winner():
            self.root.after(self.bot_speed, self._bot_turn)

    def _bot_turn(self):
        if self.bot1 is None:
            print('Error: Missing Bot')
            return
        move = self.bot1(self.env)
        self.env.step(move)
        self.drawer.draw_board()
        self.update_status()

    def start_bot_vs_bot(self):
        self.play_button.config(state="disabled")
        self.root.after(self.bot_speed, self.run_bot_vs_bot)

    def run_bot_vs_bot(self):
        if self.env.check_winner():
            self.update_status()
            return

        bot = self.bot1 if self.env.game.current_player == 1 else self.bot2
        if bot is None:
            print('Error: Missing Bot')
            return
        move = bot(self.env)
        self.env.step(move)
        self.drawer.draw_board()
        self.update_status()

        if not self.env.check_winner():
            self.root.after(self.bot_speed, self.run_bot_vs_bot)

    def update_status(self):
        winner = self.env.check_winner()
        if winner == 1:
            self.status.config(text="Player X wins!")
        elif winner == -1:
            self.status.config(text="Player O wins!")
        elif winner == 0:
            self.status.config(text="Draw!")
        else:
            turn = "X" if self.env.game.current_player == 1 else "O"
            self.status.config(text=f"Player {turn}'s turn")

    def reset(self):
        self.env.reset()
        self.drawer.draw_board()
        self.status.config(text="Player X's turn")
        self.play_button.config(state="normal")
