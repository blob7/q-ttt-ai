class CumulativeResultPlot:
    """Cumulative wins and draws, with win-rate displayed in legend."""

    def __init__(self, ax, p1_name: str, p2_name: str):
        self.ax = ax
        self.games = []
        self.p1_wins = []
        self.p2_wins = []
        self.draws = []

        self.line_p1, = ax.plot([], [], label=f"{p1_name} (WR: 0.00)")
        self.line_p2, = ax.plot([], [], label=f"{p2_name} (WR: 0.00)")
        self.line_draws, = ax.plot([], [], label="Draws")

        ax.set_xlabel("Games")
        ax.set_ylabel("Cumulative Count")
        ax.legend()

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

        self.line_p1.set_label(f"{self.line_p1.get_label().split('(')[0].strip()} (WR: {p1_wr:.2f})")
        self.line_p2.set_label(f"{self.line_p2.get_label().split('(')[0].strip()} (WR: {p2_wr:.2f})")
        self.ax.legend()

        self.ax.relim()
        self.ax.autoscale_view()
