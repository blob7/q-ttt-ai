# game/environment.py
import functools
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from .board import TicTacToe9x9, Winner
from .utils import print_board
from game.board import PlayerPiece


class GameEnv:
    """Environment wrapper for playing and training. X goes first"""
    def __init__(self):
        self.game = TicTacToe9x9()
        self.history: List[Dict[str, Any]] = [] # list of {player: PlayerPiece.value, move: (row, col), board: np.ndarray}
        self.state_history: List[Dict[str, Any]] = [] # list of {state: hash(board, current_player), action: (row, col)}
        
        # index into history that represents the current state. -1 means
        # no moves have been played (history empty). When jumping back in
        # history this will be set to that index; new moves should truncate
        # any future entries beyond this index.
        self.current_history_index = -1
        # Optional controller callables for each player. These can be set by
        # the GUI or higher-level code to allow the environment to request a
        # move for the current player (e.g. bots). Keys are PlayerPiece.X and PlayerPiece.O.
        self.controllers: dict[int, Optional[Callable]] = {PlayerPiece.X.value: None, PlayerPiece.O.value: None}

    def reset(self):
        self.game.reset()
        self.history = []
        self.state_history = []
        self.current_history_index = -1
        return self.get_state()

    def get_state(self):
        """Return a tuple or flat version of the board (suitable for AI input)."""
        return self.game.board.copy(), self.game.current_player
    
    def get_state_hash(self):
        """Return a hashable representation of the current state for use as a key in Q-tables."""
        board, player = self.get_state()
        return _cached_make_hashable(board.tobytes(), player, self.game.SIZE)

    def get_board(self):
        """Return a copy of the underlying board array (read-only from caller POV)."""
        return self.game.board.copy()

    @property
    def current_player(self):
        """Convenience property to get the current player (1 or -1)."""
        return self.game.current_player

    def step(self, action: tuple[int, int]) -> tuple[Any, bool, int | Any]:
        """Perform one move, record it, and return (state, done, winner)."""
        row, col = action
        last_state_h = self.get_state_hash()
        valid = self.game.make_move(row, col)

        if not valid:
            print(f"Invalid move: {action} by player {self.game.current_player}")
            return self.get_state(), True, None

        # Record move in history.
        # If we had previously jumped back in history (current_history_index
        # is not at the tail), truncate any "future" moves so the new move
        # continues from the selected point (undo/redo semantics).
        if self.current_history_index != len(self.history) - 1:
            # keep entries up to current_history_index (inclusive)
            self.history = self.history[: self.current_history_index + 1]
            self.state_history = self.state_history[: self.current_history_index + 1]

        # `make_move` flips current_player after applying the move, so the
        # player who just moved is the negation of the current player.
        moved_player = -self.game.current_player
        self.history.append({
            "player": moved_player,  # player who just moved (1 or -1)
            "move": (row, col),
            "board": self.game.board.copy()
        })

        self.state_history.append({
            "state": last_state_h,
            "action": (row, col)
        })

            
        # advance current_history_index to the new last entry
        self.current_history_index = len(self.history) - 1

        winner = self.evaluate_after_move()
        done = winner is not None

        return self.get_state(), done, winner

    def get_valid_moves(self):
        return self.game.get_valid_moves()

    def register_controller(self, player: int, controller: Callable):
        """Register a controller callable for a player (1 or -1).

        The controller should be a callable taking one argument (the env)
        and returning a move tuple (row, col). Use None to unregister.
        """
        if player not in (PlayerPiece.X.value, PlayerPiece.O.value):
            raise ValueError("player must be PlayerPiece.X.value or PlayerPiece.O.value")
        self.controllers[player] = controller

    def get_controller_for_player(self, player: int):
        """Return the registered controller callable for `player` or None."""
        return self.controllers.get(player)

    def get_controller_for_current_player(self):
        """Return the registered controller callable for the current player."""
        return self.get_controller_for_player(self.current_player)

    def render(self):
        """Prints the current board for debugging."""
        print_board(self.game.board)

    def check_winner(self, return_cells=False):
        return self.game.check_winner(return_cells=return_cells)
    
    def evaluate_after_move(self) -> int | Any:
        """Evaluate the game state after a move. Returns:
        - player piece if there's a winner,
        - Winner.DRAW.value if it's a draw,
        - None if ongoing.
        """
        return evaluate_after_move(
            last_player=-self.game.current_player,
            valid_moves=self.get_valid_moves(),
            last_move=self.game.last_move,
            board=self.game.board,
            turn_count=self.game.turn_count,
            win_len=self.game.win_len,
            size=self.game.SIZE
        )

    def safety_net_choices(self):
        """Return (forced_move, safe_moves)."""
        player = self.game.current_player
        opponent = self.game.last_player  # opponent is the player who just moved
        valid_moves = self.get_valid_moves()

        if self.game.turn_count <= 3 or opponent is None:
            return None, valid_moves

        # forced win
        for move in valid_moves:
            winning_move = _cached_is_winning_move(
                move=move,
                player=player,
                board_bytes=self.game.board.tobytes(),
                turn_count=self.game.turn_count + 1,
                win_len=self.game.win_len,
                game_size=self.game.SIZE
            )
            if winning_move:
                return move, valid_moves

        # moves that avoid losing next turn
        safe_moves = [
            move for move in valid_moves
            if not self.opponent_can_win_next(move, player, opponent)
        ]

        return None, safe_moves


    def opponent_can_win_next(self, move: tuple[int, int], player: int, opponent: int) -> bool:
        """Return True if applying 'move' lets opponent win on their next move."""

        # simulate our move once
        board_backup = self.game.board
        last_backup = self.game.last_move
        player_backup = self.game.current_player

        self.game.board = self._simulate_move(move, player)
        self.game.last_move = move
        self.game.current_player = opponent

        opp_moves = self.get_valid_moves()

        for opp_move in opp_moves:
            winning_move = _cached_is_winning_move(
                move=opp_move,
                player=opponent,
                board_bytes=self.game.board.tobytes(),
                turn_count=self.game.turn_count + 1,
                win_len=self.game.win_len,
                game_size=self.game.SIZE
            )
            if winning_move:
                # restore first
                self.game.board = board_backup
                self.game.last_move = last_backup
                self.game.current_player = player_backup
                return True

        # restore
        self.game.board = board_backup
        self.game.last_move = last_backup
        self.game.current_player = player_backup
        return False

    def _simulate_move(self, move: tuple[int, int], player: int) -> np.ndarray:
        """Return a copy of the board with the move applied."""
        r, c = move
        new_board = self.game.board.copy()
        new_board[r, c] = player
        return new_board
    
    def jump_to_move(self, index: int):
        """Set the game state to a specific move in history."""
        if index < 0 or index >= len(self.history):
            raise IndexError("Move index out of range")

        entry = self.history[index]
        # Restore the board snapshot and set the current player to the player
        # who should move next (i.e. the negation of the player who made
        # that recorded move).
        self.game.board = entry["board"].copy()
        self.game.current_player = -entry["player"]
        # restore last_move as well so get_valid_moves works as expected
        self.game.last_move = entry.get("move")
        # track current history index so subsequent steps truncate future
        # history entries (undo -> new move behaviour)
        self.current_history_index = index

    def export_history(self, path: str) -> None:
        """Save a minimal representation of the match history to a JSON file.

        The file contains a list of objects with 'player' and 'move' keys.
        """
        import json

        data = []
        for entry in self.history:
            mv = entry.get("move")
            data.append({"player": entry.get("player"), "move": [mv[0], mv[1]]}) # type: ignore

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load_from_file(cls, path: str) -> "GameEnv":
        """Load a match from a JSON file and return a GameEnv prepared for
        viewing. The returned env will have `history` populated and the
        board reset to the initial empty state (current_history_index == -1)
        so a viewer UI can step through moves.
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        env = cls()
        # Replay moves into env so history contains board snapshots
        for entry in data:
            mv = tuple(entry.get("move"))
            env.step(mv)

        # Reset board to initial empty state for viewing but keep the
        # populated history so the UI can step through the recorded moves.
        # Calling env.reset() would clear the history, so reset only the
        # underlying game board and track index.
        try:
            env.game.reset()
            env.current_history_index = -1
        except Exception:
            pass

        return env
    




# ----------------------------
#       cache methods
# ----------------------------
@functools.lru_cache(maxsize=2**20)
def _cached_make_hashable(board_bytes: bytes, player: int, game_size: int) -> tuple[str, int]:
    # Reconstruct the array
    board = np.frombuffer(board_bytes, dtype=int).reshape((game_size, game_size))
    
    # canonical version
    canonical_board = _canonical_board(board)
    flat = [cell for row in canonical_board for cell in row]
    encoded = ''.join(str(cell if cell >= 0 else 2) for cell in flat)
    
    return (encoded, int(player))


def _canonical_board(board: np.ndarray) -> np.ndarray:
    """
    Returns the lexicographically smallest board among all rotations and flips.
    This includes 4 rotations * (original + horizontal flip + vertical flip + both) = 16 variants.
    """
    board = np.array(board, dtype=int)  # ensure consistent type
    transforms = []
    # Generate rotations
    for k in range(4):
        rotated = np.rot90(board, k)
        transforms.append(rotated)                  # rotation only
        transforms.append(np.fliplr(rotated))      # horizontal flip
        transforms.append(np.flipud(rotated))      # vertical flip
        transforms.append(np.flipud(np.fliplr(rotated)))  # both flips
    # Pick the lexicographically smallest
    canonical = min(transforms, key=lambda b: tuple(b.flatten()))
    return canonical


@functools.lru_cache(maxsize=2**20)
def _cached_is_winning_move(board_bytes: bytes, move: tuple[int,int], player: int, game_size: int, win_len: int, turn_count: int) -> bool:
    board = np.copy(np.frombuffer(board_bytes, dtype=int).reshape((game_size, game_size)))
    r, c = move
    board[r, c] = player
    return _did_last_move_win(player, move, board, turn_count, win_len, board.shape[0])


def evaluate_after_move(last_player: int, valid_moves: list, last_move: Optional[tuple[int, int]], board: np.ndarray, turn_count: int, win_len: int, size: int) -> int | Any:
    """Return winner, draw, or None."""
    if _did_last_move_win(last_player, last_move, board, turn_count, win_len, size):
        return last_player
    if len(valid_moves) == 0:
        return Winner.DRAW.value
    return None


def _did_last_move_win(last_player: int, last_move: Optional[tuple[int, int]], board: np.ndarray, turn_count: int, win_len: int, size: int) -> bool:
    """Check only the lines that pass through self.last_move."""
    if turn_count < win_len or last_move is None:
        return False  # Not enough moves have been made to have a winner
    r, c = last_move
    b = board
    row = b[r, :]
    col = b[:, c]
    diag = b.diagonal(offset=c - r)
    anti = np.fliplr(b).diagonal(offset=(size - 1 - c) - r)
    return (
        _has_consecutive(row, last_player, win_len) or
        _has_consecutive(col, last_player, win_len) or
        _has_consecutive(diag, last_player, win_len) or
        _has_consecutive(anti, last_player, win_len)
    )

def _has_consecutive(seq, player, win_len):
    count = 0
    for x in seq:
        if x == player:
            count += 1
            if count >= win_len:
                return True
        else:
            count = 0
    return False