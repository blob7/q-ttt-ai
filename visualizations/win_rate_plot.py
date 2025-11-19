import matplotlib.pyplot as plt

class WinRatePlot:
    """Track and plot cumulative win rates for two players."""

    def __init__(self, ax):
        self.ax = ax
        self.games_played = []
        self.p1_win_rate = []
        self.p2_win_rate = []

        self.line1, = ax.plot([], [], label="Player 1")
        self.line2, = ax.plot([], [], label="Player 2")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Games Played")
        ax.set_ylabel("Win Rate")
        ax.legend()

    def update(self, game_number, p1_cum_wins, p2_cum_wins):
        self.games_played.append(game_number)
        self.p1_win_rate.append(p1_cum_wins / game_number)
        self.p2_win_rate.append(p2_cum_wins / game_number)

        self.line1.set_data(self.games_played, self.p1_win_rate)
        self.line2.set_data(self.games_played, self.p2_win_rate)
        self.ax.relim()
        self.ax.autoscale_view()
