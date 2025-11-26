class CumulativeResultPlot:
    """Cumulative wins and draws, with win-rate displayed in legend."""

    def __init__(self, ax, p1_name: str, p2_name: str):
        self.ax = ax
        self.games = []
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.p1_wins = []
        self.p2_wins = []
        self.draws = []

        self.line_p1, = ax.plot([], [], label=p1_name)
        self.line_p2, = ax.plot([], [], label=p2_name)
        self.line_draws, = ax.plot([], [], label="Draws")

        self.caption = None
        self.ax.figure.subplots_adjust(bottom=0.2)

        self._display_labels(0, 0)

    def update(self, p1_win: bool, p2_win: bool, draw: bool):
        self.games.append(self.games[-1] + 1 if self.games else 1)

        self.p1_wins.append((self.p1_wins[-1] if self.p1_wins else 0) + int(p1_win))
        self.p2_wins.append((self.p2_wins[-1] if self.p2_wins else 0) + int(p2_win))
        self.draws.append((self.draws[-1] if self.draws else 0) + int(draw))

        # Update lines
        self.line_p1.set_data(self.games, self.p1_wins)
        self.line_p2.set_data(self.games, self.p2_wins)
        self.line_draws.set_data(self.games, self.draws)

        # Update legend text with win rate
        total_games = self.games[-1] if self.games else 0
        p1_wr = self.p1_wins[-1] / total_games if total_games > 0 else 0
        p2_wr = self.p2_wins[-1] / total_games if total_games > 0 else 0

        self._display_labels(p1_wr, p2_wr)

        self.ax.relim()
        self.ax.autoscale_view()

    def _display_labels(self, p1_wr: float, p2_wr: float):
        self.ax.set_xlabel("Games")
        self.ax.set_ylabel("Cumulative Count")
        self.ax.set_title("Cumulative Results Over Time")
        if self.caption is None:
            self.caption = self.ax.text(
                0.5, -0.2,
                f"{self.p1_name} WR: {p1_wr:.2f} \n {self.p2_name} WR: {p2_wr:.2f}",
                ha="center",
                va="top",
                transform=self.ax.transAxes,
            )
        else:
            self.caption.set_text(f"{self.p1_name} WR: {p1_wr:.2f}% \n {self.p2_name} WR: {p2_wr:.2f}%")
        self.ax.legend()