import tkinter as tk
from typing import Optional, Callable, Literal

from gui.drawer import BoardDrawer
from gui.controller import GameController
from gamemodes import GameMode
from game.environment import GameEnv
from game.board import Winner, PlayerPiece
from gui.components import MoveHistoryPanel


class TicTacToeGUI:
    def __init__(self, mode: GameMode, bot1: Optional[Callable] = None, bot2: Optional[Callable] = None):
        self.env: GameEnv = GameEnv()
        self.mode: GameMode = mode

        # --- UI Setup (View builds widgets) ---
        self.root = tk.Tk()
        self.root.title("9x9 Directional Tic Tac Toe")

        # Set a sensible initial window size so the history panel doesn't
        # overlap the board on small displays. Compute width from board +
        # history widths and a small margin.
        board_px = 9 * 50
        history_px = 200
        margin = 220
        init_w = board_px + history_px + margin
        init_h = max(9 * 50 + 120, 520)
        try:
            self.root.geometry(f"{init_w}x{init_h}")
        except Exception:
            pass

        # Drawer (widget-style: BoardDrawer is a Frame that owns its Canvas)
        self.drawer = BoardDrawer(self.root, self.env, cells=9, cell_size=50)
        # place the drawer Frame in the layout (only in column 0 so the
        # move history panel can occupy column 1 without overlap)
        # Add a small padding so the board doesn't touch the window edges.
        self.drawer.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=6, pady=6)

        # Configure root layout weights (UI concern kept in the View)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=0)
        self.root.grid_columnconfigure(3, weight=0)

        # Move history panel (created without on_select_move; controller will wire callback)
        self.move_panel = MoveHistoryPanel(self.root, env=self.env, on_select_move=None, width=200, height=400)
        self.move_panel.grid(row=0, column=1, rowspan=4, sticky="ns", padx=5, pady=5)

        # Save match button (available in play modes). Hidden in VIEW_MATCH.
        self.save_button = None
        if self.mode != GameMode.VIEW_MATCH:
            self.save_button = tk.Button(self.root, text="Save Match")
            self.save_button.grid(row=5, column=1, sticky="ew", padx=5, pady=2)
        # Load match only shown in VIEW_MATCH mode (option 5)
        self.load_button = None
        if self.mode == GameMode.VIEW_MATCH:
            self.load_button = tk.Button(self.root, text="Load Match")
            self.load_button.grid(row=5, column=1, sticky="ew", padx=5, pady=2)

        # Status label
        self.status_label = tk.Label(self.root, text="Player X's turn", font=("Arial", 14))
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)

        # Control buttons
        self.reset_button = tk.Button(self.root, text="Reset")
        self.reset_button.grid(row=4, column=2, sticky="ew", padx=2)
        self.quit_button = tk.Button(self.root, text="Quit")
        self.quit_button.grid(row=5, column=2, sticky="ew", padx=2, pady=2)

        # Speed input and play button placeholders
        self.speed_input = None
        self.play_button = None
        if self.mode in (GameMode.PLAYER_V_BOT, GameMode.BOT_V_BOT, GameMode.VIEW_MATCH):
            from gui.components import SpeedInput
            self.speed_input = SpeedInput(self.root, default_speed=500, on_change=None)
            # Place speed control in its own column so it doesn't overlap
            # the history Save/Load buttons or the Quit button.
            self.speed_input.grid(row=5, column=3, sticky="ew", pady=2)

        # Provide a play/pause toggle for both Player-v-Bot and Bot-v-Bot
        if self.mode in (GameMode.PLAYER_V_BOT, GameMode.BOT_V_BOT, GameMode.VIEW_MATCH):
            from gui.components import PlayPauseButton
            # Controller will wire the callbacks via attach_view
            self.play_button = PlayPauseButton(self.root, on_play=None, on_pause=None, initial_playing=False)
            self.play_button.grid(row=5, column=0, sticky="ew", pady=2)

        # Create controller after view elements exist; controller will attach callbacks
        self.controller = GameController(self.root, self.env, mode, bot1, bot2)
        self.controller.attach_view(self)

        # Initial draw
        self.drawer.draw_board()

    def run(self):
        self.root.mainloop()

    # --- View refresh helpers ---
    def refresh_board(self):
        """Ask the drawer to redraw itself from the env state."""
        self.drawer.draw_board()

    def refresh_status(self):
        """Update the status label from the env state."""
        winner = self.env.check_winner()
        if winner == Winner.X.value:
            text = "Player X wins!"
        elif winner == Winner.O.value:
            text = "Player O wins!"
        elif winner == Winner.DRAW.value:
            text = "Draw!"
        else:
            turn = "X" if self.env.current_player == PlayerPiece.X.value else "O"
            text = f"Player {turn}'s turn"
        self.status_label.config(text=text)

    def refresh_history(self):
        """Tell the move history panel to refresh itself from the env."""
        # MoveHistoryPanel reads `env.history` so just tell it to update.
        self.move_panel.update_history(current_index=self.env.current_history_index)

    def refresh_ui(self, set_play_state: Optional[Literal['normal', 'active', 'disabled']] = None):
        """Convenience helper to refresh board, status and history in one call.

        set_play_state: if provided, will be passed to `set_play_button_state`.
        """
        try:
            self.refresh_board()
            self.refresh_status()
            self.refresh_history()
        except Exception:
            print("Error refreshing UI components")
        if set_play_state is not None:
            try:
                self.set_play_button_state(set_play_state)
            except Exception:
                pass

    def set_play_button_state(self, state: Literal['normal', 'active', 'disabled']):
        """Set the play button state if it exists (UI-only helper)."""
        if self.play_button is not None:
            try:
                self.play_button.config(state=state)
            except Exception:
                print("Error setting play button state")
