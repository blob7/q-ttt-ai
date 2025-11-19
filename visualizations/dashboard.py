import matplotlib.pyplot as plt

from game.board import PlayerPiece

from .win_rate_plot import WinRatePlot
from .heatmap_plot import MoveHeatmap
from .game_length_plot import GameLengthPlot

class GameDashboard:
    """Combines all plots in a single dashboard window."""

    def __init__(self, p1_name: str = "Player 1", p2_name: str = "Player 2", board_size: int = 9):
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.game_number = 0

        # Initialize wins
        if p1_name == p2_name:
            self.wins = {f"{p1_name} - X": 0, f"{p1_name} - O": 0}
        else:
            self.wins = {p1_name: 0, p2_name: 0}

        # Create 2x2 figure (last subplot empty)
        self.fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[1, 1].axis('off')

        # Initialize plot modules
        self.win_rate_plot = WinRatePlot(axs[0, 0])
        self.heatmap_plot = MoveHeatmap(axs[0, 1], board_size)
        self.game_length_plot = GameLengthPlot(axs[1, 0])

        plt.ion()
        plt.show()

    def update(self, stats):
        """Update dashboard with new game data."""
        self.game_number += 1

        winner = stats['winning_player']      # winner name
        piece = stats.get('winning_piece')    # 'X' or 'O' for winning piece

        # Determine correct key in wins dict
        if self.p1_name == self.p2_name:
            if not piece:
                raise ValueError("Must pass 'piece' when both players have the same name")
            piece_symbol = 'X' if piece == PlayerPiece.X.value else 'O'
            winner_key = f"{winner} - {piece_symbol}"
        else:
            winner_key = winner

        # Update wins
        if winner_key in self.wins:
            self.wins[winner_key] += 1
        else:
            raise ValueError(f"Unknown winner '{winner_key}' in wins dict")

        # For plotting: extract totals per player
        if self.p1_name == self.p2_name:
            p1_wins = self.wins[f"{self.p1_name} - X"]
            p2_wins = self.wins[f"{self.p2_name} - O"]
        else:
            p1_wins = self.wins[self.p1_name]
            p2_wins = self.wins[self.p2_name]

        # Update plots
        self.win_rate_plot.update(self.game_number, p1_wins, p2_wins)
        self.heatmap_plot.update(stats['last_move'])
        self.game_length_plot.update(stats['turns'])

        plt.pause(0.01)

