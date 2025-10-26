# game/environment.py
from .board import TicTacToe9x9
from .utils import print_board


class GameEnv:
    """Environment wrapper for playing and training."""
    def __init__(self):
        self.game = TicTacToe9x9()
        self.history = []

    def reset(self):
        self.game.reset()
        self.history = []
        return self.get_state()

    def get_state(self):
        """Return a tuple or flat version of the board (suitable for AI input)."""
        return self.game.board.copy(), self.game.current_player

    def step(self, action: tuple[int, int]):
        """Perform one move, record it, and return (state, reward, done, winner)."""
        row, col = action
        valid = self.game.make_move(row, col)

        if not valid:
            # Invalid move: penalize
            return self.get_state(), -1, True, None

        # Record move in history
        self.history.append({
            "player": self.game.current_player * -1,  # player who just moved
            "move": (row, col),
            "board": self.game.board.copy()
        })

        winner = self.check_winner()
        done = winner is not None

        if winner == self.game.current_player * -1:
            reward = 1  # last move wins
        elif winner == 0:
            reward = 0.5  # draw
        elif not done:
            reward = 0
        else:
            reward = -1  # lost (rare)

        return self.get_state(), reward, done, winner

    def get_valid_moves(self):
        return self.game.get_valid_moves()

    def render(self):
        """Prints the current board for debugging."""
        print_board(self.game.board)

    def check_winner(self, return_cells=False):
        return self.game.check_winner(return_cells=return_cells)