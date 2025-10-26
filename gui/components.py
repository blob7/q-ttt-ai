import tkinter as tk


class SpeedInput(tk.Frame):
    """A simple labeled text input for bot move speed (in ms)."""
    def __init__(self, master, default_speed=500, on_change=None):
        super().__init__(master)
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
    def __init__(self, master, on_click):
        super().__init__(master, text="Play", command=on_click)
