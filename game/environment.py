# game/environment.py
from .board import TicTacToe9x9

class GameEnv:
    """Environment wrapper for playing and training."""
    def __init__(self):
        self.game = TicTacToe9x9()

    def reset(self):
        self.game.reset()
        return self.get_state()

    def get_state(self):
        """Return a tuple or flat version of the board (suitable for AI input)."""
        return self.game.board.copy(), self.game.current_player

    def step(self, action):
        """Perform one move and return (next_state, reward, done, winner)."""
        row, col = action
        valid = self.game.make_move(row, col)
        if not valid:
            # Invalid move (penalize)
            return self.get_state(), -1, True, None

        winner = self.game.check_winner()
        done = winner is not None

        if winner == self.game.current_player * -1:
            reward = 1   # the move just made wins
        elif winner == 0:
            reward = 0.5  # draw
        elif not done:
            reward = 0
        else:
            reward = -1   # lost (should rarely trigger this way)

        return self.get_state(), reward, done, winner

    def get_valid_moves(self):
        return self.game.get_valid_moves()

    def render(self):
        """Prints the current board for debugging."""
        from .utils import print_board
        print_board(self.game.board)
