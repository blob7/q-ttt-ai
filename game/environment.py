# game/environment.py
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from .board import TicTacToe9x9, Winner
from .utils import print_board
from game.board import PlayerPiece
from game.shared_cache import digest_bytes, get_cache


Coord = Tuple[int, int]
Board = np.ndarray
Transform = Callable[[Board], Board]
MoveTransform = Callable[[Coord, int], Coord]
Line = List[Coord]

def precompute_lines(size: int, win_len: int) -> dict[tuple[int, int], list[list[tuple[int, int]]]]:
    """Precompute all win_len-length lines that contain each cell."""
    mapping: dict[tuple[int, int], list[list[tuple[int, int]]]] = {
        (r, c): [] for r in range(size) for c in range(size)
    }

    dirs = [
        (0, 1),   # horizontal →
        (1, 0),   # vertical ↓
        (1, 1),   # diag ↘
        (1, -1),  # anti-diag ↙
    ]

    for r in range(size):
        for c in range(size):
            for dr, dc in dirs:
                # Try to build a line centered around (r,c)
                for offset in range(-(win_len - 1), 1):
                    line: list[tuple[int, int]] = []
                    for k in range(win_len):
                        rr = r + (offset + k) * dr
                        cc = c + (offset + k) * dc
                        if 0 <= rr < size and 0 <= cc < size:
                            line.append((rr, cc))
                        else:
                            break
                    if len(line) == win_len and (r, c) in line:
                        mapping[(r, c)].append(line)

    return mapping

LINES_BY_CELL = precompute_lines(9, 3)


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
        return self.game.board.copy(), self.game.current_player, self.game.last_move
    
    def get_state_hash(self):
        """Return a hashable representation of the current state for use as a key in Q-tables."""
        board, player, last_move = self.get_state()
        return _cached_make_hashable(board.tobytes(), player, self.game.SIZE, last_move)

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
        """scans for a winner"""
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
        """Return (forced_move, safe_moves). Uses precomputed lines to detect forced wins/blocks."""
        player = self.game.current_player
        opponent = self.game.last_player
        valid_moves = self.get_valid_moves()

        if self.game.turn_count < self.game.win_len or opponent is None:
            return None, valid_moves

        b = self.game.board
        # ---------- Forced win (fast, exact) ----------
        for mv in valid_moves:
            for line in LINES_BY_CELL[mv]:
                if _line_becomes_win(b, line, mv, player):
                    # immediate win found
                    return mv, valid_moves

        # ---------- Safe moves ----------
        safe_moves = []
        for mv in valid_moves:
            if not self.opponent_can_win_next(mv, player, opponent):
                safe_moves.append(mv)

        return None, safe_moves




    # def opponent_can_win_next(self, move: tuple[int, int], player: int, opponent: int) -> bool:
    #     """Return True if applying 'move' lets opponent win on their next move."""

    #     # simulate our move once
    #     board_backup = self.game.board
    #     last_backup = self.game.last_move
    #     player_backup = self.game.current_player

    #     self.game.board = self._simulate_move(move, player)
    #     self.game.last_move = move
    #     self.game.current_player = opponent

    #     opp_moves = self.get_valid_moves()

    #     for opp_move in opp_moves:
    #         winning_move = _cached_is_winning_move(
    #             move=opp_move,
    #             player=opponent,
    #             board_bytes=self.game.board.tobytes(),
    #             turn_count=self.game.turn_count + 1,
    #             win_len=self.game.win_len,
    #             game_size=self.game.SIZE
    #         )
    #         if winning_move:
    #             # restore first
    #             self.game.board = board_backup
    #             self.game.last_move = last_backup
    #             self.game.current_player = player_backup
    #             return True

    #     # restore
    #     self.game.board = board_backup
    #     self.game.last_move = last_backup
    #     self.game.current_player = player_backup
    #     return False

    # def opponent_can_win_next(
    #     self,
    #     move: tuple[int, int],
    #     player: int,
    #     opponent: int
    # ) -> bool:
    #     """Return True if applying 'move' lets opponent immediately win next turn."""

    #     board_backup = self.game.board
    #     last_backup = self.game.last_move
    #     player_backup = self.game.current_player

    #     self.game.board = self._simulate_move(move, player)
    #     self.game.last_move = move
    #     self.game.current_player = opponent

    #     opp_moves = self.get_valid_moves()
    #     b = self.game.board

    #     for om in opp_moves:
    #         # For each line that contains this move
    #         for line in LINES_BY_CELL[om]:
    #             # Simulate opponent placing om
    #             # We only modify one cell temporarily
    #             if b[om] != 0:
    #                 continue  # impossible but safe

    #             # temporarily apply move
    #             b[om] = opponent
    #             if line_would_win(b, line, opponent):
    #                 b[om] = 0
    #                 self.game.board = board_backup
    #                 self.game.last_move = last_backup
    #                 self.game.current_player = player_backup
    #                 return True
    #             # undo
    #             b[om] = 0

    #     self.game.board = board_backup
    #     self.game.last_move = last_backup
    #     self.game.current_player = player_backup
    #     return False


    def opponent_can_win_next(self, move: tuple[int,int], player: int, opponent: int) -> bool:
        """
        Return True if applying 'move' (by player) allows 'opponent' to immediately win next turn.

        This applies the move in-place, temporarily adjusts game state so get_valid_moves()
        returns the opponent's legal moves, then checks only the precomputed win-length
        lines that pass through each opponent candidate cell.
        """
        b = self.game.board
        r, c = move

        # Apply move in-place
        prev = b[r, c]
        b[r, c] = player

        # Temporarily adjust game metadata used by get_valid_moves()
        last_backup = self.game.last_move
        current_backup = self.game.current_player
        self.game.last_move = move
        self.game.current_player = opponent

        try:
            opp_moves = self.get_valid_moves()  # valid under the simulated board/last_move

            for om in opp_moves:
                # skip if cell already occupied (defensive)
                if b[om] != 0:
                    continue

                # For each win_len-sized line that contains this opponent move
                for line in LINES_BY_CELL[om]:
                    # Quick prune: if any other cell in the line is occupied by 'player', can't win on this line
                    blocked = False
                    for rr, cc in line:
                        if (rr, cc) == om:
                            continue
                        if b[rr, cc] == player:
                            blocked = True
                            break
                    if blocked:
                        continue

                    # Check if all other cells are opponent -> immediate win if opponent places at om
                    win = True
                    for rr, cc in line:
                        if (rr, cc) == om:
                            continue
                        if b[rr, cc] != opponent:
                            win = False
                            break
                    if win:
                        return True

            return False
        finally:
            # Restore board and metadata
            b[r, c] = prev
            self.game.last_move = last_backup
            self.game.current_player = current_backup



    # def _simulate_move(self, move: tuple[int, int], player: int) -> np.ndarray:
    #     """Return a copy of the board with the move applied."""
    #     r, c = move
    #     new_board = self.game.board.copy()
    #     new_board[r, c] = player
    #     return new_board
    
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
def _cached_make_hashable(
    board_bytes: bytes,
    player: int,
    game_size: int,
    last_move: Optional[Coord]
) -> Tuple[str, int, Coord]:
    cache = get_cache("state_hash")
    board_digest = digest_bytes(board_bytes)
    move_key = last_move if last_move is not None else (-1, -1)

    def compute() -> Tuple[str, int, Coord]:
        board = np.frombuffer(board_bytes, dtype=int).reshape((game_size, game_size))
        canonical_board, canonical_move = canonicalize_board_and_move(board, last_move)
        flat = "".join(str(int(x) if x >= 0 else 2) for x in canonical_board.flatten())
        lm = canonical_move if canonical_move is not None else (-1, -1)
        return (flat, player, lm)

    return cache.get_or_set((board_digest, player, game_size, move_key), compute)



def canonicalize_board_and_move(
    board: Board,
    last_move: Optional[Coord]
) -> Tuple[Board, Optional[Coord]]:
    """Return canonical board and move using D4 symmetries."""
    
    size = board.shape[0]

    best_key = None
    best_board = None
    best_move = None

    for board_tf, move_tf in d4_transforms(size):
        
        b2 = board_tf(board)

        # Flatten key used for lexicographic comparison
        key = tuple(int(x) for x in b2.flatten())
        
        if best_key is None or key < best_key:
            best_key = key
            best_board = b2

            if last_move is not None:
                best_move = move_tf(last_move)
            else:
                best_move = None

    assert best_board is not None
    return best_board, best_move



def d4_transforms(size: int):
    """Return 8 board and move transforms of the D4 symmetry group."""
    
    def rotate_move(coord: Coord, k: int) -> Coord:
        r, c = coord
        for _ in range(k % 4):
            r, c = size - 1 - c, r
        return r, c

    def flip_lr_move(coord: Coord) -> Coord:
        r, c = coord
        return r, size - 1 - c

    def flip_ud_move(coord: Coord) -> Coord:
        r, c = coord
        return size - 1 - r, c

    # 1. Identity
    yield (lambda b: b, lambda m: m)

    # 2–4. Rot90, Rot180, Rot270
    for k in (1, 2, 3):
        yield (lambda b, k=k: rot90(b, k), lambda m, k=k: rotate_move(m, k))

    # 5. Reflect (vertical flip)
    yield (flip_lr, lambda m: flip_lr_move(m))

    # 6. Reflect (horizontal flip)
    yield (flip_ud, lambda m: flip_ud_move(m))

    # 7. Reflect across main diagonal (transpose)
    yield (lambda b: b.T, lambda m: (m[1], m[0]))

    # 8. Reflect across anti-diagonal
    yield (lambda b: rot90(b, 1).T, lambda m: rotate_move((m[1], m[0]), 3))


def rot90(b: Board, k: int) -> Board:
    return np.rot90(b, k)

def flip_lr(b: Board) -> Board:
    return np.fliplr(b)

def flip_ud(b: Board) -> Board:
    return np.flipud(b)

def _cached_is_winning_move(
    board_bytes: bytes,
    move: tuple[int,int],
    player: int,
    game_size: int,
    win_len: int,
    turn_count: int
) -> bool:
    cache = get_cache("winning_move")
    board_digest = digest_bytes(board_bytes)

    def compute() -> bool:
        board = np.frombuffer(board_bytes, dtype=int).reshape((game_size, game_size)).copy()
        board[move] = player
        return _did_last_move_win(
            last_player=player,
            last_move=move,
            board=board,
            turn_count=turn_count,
            win_len=win_len,
            size=game_size,
            lines_by_cell=LINES_BY_CELL,
        )

    key = (board_digest, move, player, game_size, win_len, turn_count)
    return cache.get_or_set(key, compute)



def evaluate_after_move(
    last_player: int,
    valid_moves: list,
    last_move: tuple[int,int] | None,
    board: np.ndarray,
    turn_count: int,
    win_len: int,
    size: int,
):
    if _did_last_move_win(last_player, last_move, board, turn_count, win_len, size, LINES_BY_CELL):
        return last_player
    if not valid_moves:
        return Winner.DRAW.value
    return None



def _did_last_move_win(
    last_player: int,
    last_move: tuple[int, int] | None,
    board: np.ndarray,
    turn_count: int,
    win_len: int,
    size: int,
    lines_by_cell: dict[tuple[int,int], list[list[tuple[int,int]]]]
) -> bool:
    if last_move is None or turn_count < win_len:
        return False

    r, c = last_move

    # All lines through this cell
    for line in lines_by_cell[(r, c)]:
        count = 0
        for rr, cc in line:
            if board[rr, cc] == last_player:
                count += 1
                if count == win_len:
                    return True
            else:
                count = 0
    return False




def line_would_win(board: np.ndarray, line: list[tuple[int, int]], player: int) -> bool:
    """Check if placing a move on the last cell of the line forms a win."""
    # All cells must match the player.
    for rr, cc in line:
        if board[rr, cc] != player:
            return False
    return True

def _line_becomes_win(board: np.ndarray, line: Line, cell: Coord, player: int) -> bool:
    """Return True if placing player at `cell` completes this `line`."""
    # All other cells in the line must already be occupied by `player`.
    for rr, cc in line:
        if (rr, cc) == cell:
            continue
        if board[rr, cc] != player:
            return False
    return True