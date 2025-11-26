from matplotlib import ticker

class GameLengthPlot:
    """Histogram of turns per game."""

    def __init__(self, ax):
        self.ax = ax
        self.turns_list = []
        self._ensure_margin()
        self._redraw()

    def update(self, num_turns: int):
        self.turns_list.append(num_turns)
        self._redraw()

    def _redraw(self) -> None:
        self.ax.cla()
        max_count = 0
        if self.turns_list:
            counts, _, _ = self.ax.hist(self.turns_list, bins=range(1, max(self.turns_list) + 2))
            max_count = int(max(counts)) if counts.size > 0 else 0
        self.ax.set_xlabel("Turns per Game")
        self.ax.set_ylabel("Count")
        self.ax.set_title("Game Length Distribution")
        upper = max_count if max_count > 0 else 1
        self.ax.set_ylim(0, upper)
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    def _ensure_margin(self) -> None:
        fig = self.ax.figure
        if fig is None:
            return
        if fig.subplotpars.bottom < 0.2:
            fig.subplots_adjust(bottom=0.2)