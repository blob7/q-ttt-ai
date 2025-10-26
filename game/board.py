# game/board.py
import numpy as np

class TicTacToe9x9:
    """9x9 Tic Tac Toe with directional move restriction and 3-in-a-row win."""
    SIZE = 9

    def __init__(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)  # 0 = empty, 1 = X, -1 = O
        self.current_player = 1
        self.last_move = None

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.last_move = None

    def get_valid_moves(self):
        if self.last_move is None:
            # First move can go anywhere
            return [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)]
        
        r, c = self.last_move
        valid = set()

        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while (
                0 <= nr < self.SIZE
                and 0 <= nc < self.SIZE
                and self.board[nr, nc] == 0
            ):
                valid.add((nr, nc))
                nr += dr
                nc += dc

        return list(valid)

    def make_move(self, row, col):
        if (row, col) not in self.get_valid_moves():
            return False  # invalid move
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.current_player *= -1
        return True

    def check_winner(self, return_cells=False):
        """Returns:
        - 1 if X wins
        - -1 if O wins
        - 0 if draw
        - None if ongoing

        If return_cells=True, also returns the list of winning cell coordinates.
        """
        b = self.board
        size = self.SIZE

        for r in range(size):
            for c in range(size):
                if b[r, c] == 0:
                    continue
                player = b[r, c]
                for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    cells = [(r + i*dr, c + i*dc) for i in range(3)]
                    if all(
                        0 <= rr < size and 0 <= cc < size and b[rr, cc] == player
                        for rr, cc in cells
                    ):
                        return (player, cells) if return_cells else player

        if not np.any(b == 0):
            return (0, None) if return_cells else 0
        return (None, None) if return_cells else None