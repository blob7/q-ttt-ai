import matplotlib.pyplot as plt

class WinRatePlot:
    """Track and plot cumulative win rates for two players."""

    def __init__(self):
        self.games_played = []
        self.player1_win_rate = []
        self.player2_win_rate = []

        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], label="Player 1")
        self.line2, = self.ax.plot([], [], label="Player 2")
        self.ax.set_xlabel("Games Played")
        self.ax.set_ylabel("Win Rate")
        self.ax.set_ylim(0, 1)
        self.ax.legend()
        plt.ion()
        plt.show()

    def update(self, game_number, p1_cum_wins, p2_cum_wins):
        self.games_played.append(game_number)
        self.player1_win_rate.append(p1_cum_wins / game_number)
        self.player2_win_rate.append(p2_cum_wins / game_number)

        self.line1.set_data(self.games_played, self.player1_win_rate)
        self.line2.set_data(self.games_played, self.player2_win_rate)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)  # Small pause to see live-ish update
