import tkinter as tk

from game.board import PlayerPiece

class SpeedInput(tk.Frame):
    """A simple labeled text input for bot move speed (in ms)."""
    def __init__(self, parent, default_speed=500, on_change=None):
        super().__init__(parent)
        self.on_change = on_change
        self.var = tk.StringVar(value=str(default_speed))

        tk.Label(self, text="Speed (ms):").pack(side="left", padx=3)
        entry = tk.Entry(self, textvariable=self.var, width=6)
        entry.pack(side="left")
        entry.bind("<Return>", self._update_speed)
        entry.bind("<FocusOut>", self._update_speed)

    def _update_speed(self, event=None):
        try:
            value = int(self.var.get())
            if value < 50:
                value = 50
            if self.on_change:
                self.on_change(value)
        except ValueError:
            pass  # ignore invalid input


class PlayButton(tk.Button):
    """A simple Play button for Bot vs Bot mode."""
    def __init__(self, parent, on_click):
        super().__init__(parent, text="Play", command=on_click)


class PlayPauseButton(tk.Button):
    """Toggle-style Play/Pause button. Call `set_playing(True|False)` to
    update state programmatically.
    """
    def __init__(self, parent, on_play, on_pause, initial_playing: bool = False):
        self.on_play = on_play
        self.on_pause = on_pause
        self._playing = bool(initial_playing)
        text = "Pause" if self._playing else "Play"
        super().__init__(parent, text=text, command=self._toggle)

    def _toggle(self):
        if self._playing:
            self.set_playing(False)
            try:
                if self.on_pause:
                    self.on_pause()
            except Exception:
                pass
        else:
            self.set_playing(True)
            try:
                if self.on_play:
                    self.on_play()
            except Exception:
                pass

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        try:
            self.config(text="Pause" if self._playing else "Play")
        except Exception:
            pass


class MoveHistoryPanel(tk.Frame):
    def __init__(self, parent, env, on_select_move=None, width=200, height=400):
        """
        parent: parent Tkinter widget
        env: GameEnv instance
        on_select_move: optional callback (move_index) when a move is clicked
        """
        super().__init__(parent, width=width, height=height, bd=2, relief="groove", bg="#f5f5f5")
        self.env = env
        self.on_select_move = on_select_move
        self.current_index = -1  # index of currently highlighted move
        self.select_highlight_color = "#dddddd"

        # Canvas with scrollbar for move list
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner_frame = tk.Frame(self.canvas)

        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.labels = []

    def update_history(self, current_index: int | None = None):
        """Refresh move history from env.history."""
        for lbl in self.labels:
            lbl.destroy()
        self.labels.clear()

        if not self.env.history:
            return  # nothing to display yet

        self.current_index = current_index

        for i, entry in enumerate(self.env.history):
            player = "X" if entry["player"] == PlayerPiece.X.value else "O"
            move = entry["move"]
            text = f"{i+1}. {player} → {move}"

            bg = self.select_highlight_color if i == self.current_index else None

            lbl = tk.Label(self.inner_frame, text=text, anchor="w", bg=bg) # type: ignore
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))

            self.labels.append(lbl)


    def _on_click(self, index):
        if self.on_select_move:
            self.on_select_move(index)