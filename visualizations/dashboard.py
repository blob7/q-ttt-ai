import matplotlib.pyplot as plt

from game.board import PlayerPiece

from .cumulative_results_plot import CumulativeResultPlot
from .xo_breakdown import PieceBreakdownPlot
from .heatmap_plot import MoveHeatmap
from .game_length_plot import GameLengthPlot

class GameDashboard:
    """Combines all plots in a single dashboard window."""

    def __init__(self, p1_name: str = "Player 1", p2_name: str = "Player 2", board_size: int = 9):
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.game_number = 0

        # Initialize wins
        self.display_name1 = f"{p1_name} - X" if p1_name == p2_name else p1_name
        self.display_name2 = f"{p2_name} - O" if p1_name == p2_name else p2_name

        # Create 2x2 figure (last subplot empty)
        self.fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[1, 1].axis('off')

        title_text = f"{self.display_name1} vs {self.display_name2} Dashboard"
        self.fig.suptitle(title_text, fontsize=18, fontweight="bold", y=0.98)

        self.subtitle = self.fig.text(
            0.5, 0.95, "Game 0",
            ha='center', va='center', fontsize=12
        )

        # Initialize plot modules
        self.cumulative_result_plot = CumulativeResultPlot(axs[0, 0], self.display_name1, self.display_name2)
        self.piece_breakdown_plot = PieceBreakdownPlot(axs[1, 0], self.display_name1, self.display_name2)
        self.heatmap_plot = MoveHeatmap(axs[0, 1], board_size)
        self.game_length_plot = GameLengthPlot(axs[1, 1])

        plt.ion()
        plt.show()

    def finalize(self):
        plt.ioff()
        plt.show(block=True)


    def update(self, stats):
        """Update dashboard with new game data."""
        self.game_number += 1
        self.subtitle.set_text(f"Game {self.game_number}")

        winner = stats['winning_player']      # winner name
        piece = stats.get('winning_piece')    # 'X' or 'O' for winning piece

        # Determine correct winner name
        winner_key = winner
        if self.p1_name == self.p2_name:
            if piece == PlayerPiece.X.value:
                winner_key = f"{winner} - X"
            elif piece == PlayerPiece.O.value:
                winner_key = f"{winner} - O"
            else:
                winner_key = None

        p1_win = winner_key == self.display_name1
        p1_won_as_x = p1_win and piece == PlayerPiece.X.value
        p1_won_as_o = p1_win and piece == PlayerPiece.O.value

        p2_win = winner_key == self.display_name2
        p2_won_as_x = p2_win and piece == PlayerPiece.X.value
        p2_won_as_o = p2_win and piece == PlayerPiece.O.value

        draw = winner_key is None

        # Update plots
        self.heatmap_plot.update(stats['last_move'])
        self.cumulative_result_plot.update(p1_win, p2_win, draw)
        self.game_length_plot.update(stats['turns'])
        self.piece_breakdown_plot.update(p1_won_as_x, p1_won_as_o, p2_won_as_x, p2_won_as_o, draw)

        plt.pause(0.01)