# game/board.py
from typing import Any
import numpy as np
from enum import Enum



class PlayerPiece(Enum):
    X = 1
    O = -1

class Winner(Enum):
    X = PlayerPiece.X.value
    O = PlayerPiece.O.value
    DRAW = 0
    ONGOING = None

class TicTacToe9x9:
    """9x9 Tic Tac Toe with directional move restriction and 3-in-a-row win."""

    def __init__(self):
        self.SIZE = 9
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)  # 0 = empty, 1 = X, -1 = O
        self.win_len = 3
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.current_player = PlayerPiece.X.value
        self.last_player = None
        self.last_move = None
        self.turn_count = 0
        

    def get_valid_moves(self):
        return _compute_valid_moves(self.board, self.last_move, self.SIZE)

    def make_move(self, row: int, col: int):
        if (row, col) not in self.get_valid_moves():
            return False  # invalid move
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.turn_count += 1
        # Flip current player using the PlayerPiece enum values rather than
        # hard-coded 1/-1.
        if self.current_player == PlayerPiece.X.value:
            self.current_player = PlayerPiece.O.value
            self.last_player = PlayerPiece.X.value
        else:
            self.current_player = PlayerPiece.X.value
            self.last_player = PlayerPiece.O.value
        return True

    def check_winner(self, return_cells: bool = False) -> Any:
        """Returns:
        - None if ongoing or player piece if there's a winner, or 0 for draw.

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
            return (Winner.DRAW.value, None) if return_cells else Winner.DRAW.value
        return (Winner.ONGOING.value, None) if return_cells else Winner.ONGOING.value

    
def _compute_valid_moves(
    board: np.ndarray,
    last_move: tuple[int, int] | None,
    game_size: int,
) -> list[tuple[int, int]]:
    if last_move is None:
        return [(r, c) for r in range(game_size) for c in range(game_size) if board[r, c] == 0]

    r, c = last_move
    valid: set[tuple[int, int]] = set()

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while 0 <= nr < game_size and 0 <= nc < game_size and board[nr, nc] == 0:
            valid.add((nr, nc))
            nr += dr
            nc += dc

    return sorted(valid)