# game/environment.py
from typing import Any, Callable, Optional
import numpy as np
from .board import TicTacToe9x9
from .utils import print_board
from game.board import PlayerPiece


class GameEnv:
    """Environment wrapper for playing and training."""
    def __init__(self):
        self.game = TicTacToe9x9()
        self.history = []
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
        self.current_history_index = -1
        return self.get_state()

    def get_state(self):
        """Return a tuple or flat version of the board (suitable for AI input)."""
        return self.game.board.copy(), self.game.current_player

    def get_board(self):
        """Return a copy of the underlying board array (read-only from caller POV)."""
        return self.game.board.copy()

    @property
    def current_player(self):
        """Convenience property to get the current player (1 or -1)."""
        return self.game.current_player

    def step(self, action: tuple[int, int]) -> tuple[Any, bool, int | Any]:
        """Perform one move, record it, and return (state, reward, done, winner)."""
        row, col = action
        valid = self.game.make_move(row, col)

        if not valid:
            # Invalid move: penalize
            return self.get_state(), True, None

        # Record move in history.
        # If we had previously jumped back in history (current_history_index
        # is not at the tail), truncate any "future" moves so the new move
        # continues from the selected point (undo/redo semantics).
        if self.current_history_index != len(self.history) - 1:
            # keep entries up to current_history_index (inclusive)
            self.history = self.history[: self.current_history_index + 1]

        # `make_move` flips current_player after applying the move, so the
        # player who just moved is the negation of the current player.
        moved_player = -self.game.current_player
        self.history.append({
            "player": moved_player,  # player who just moved (1 or -1)
            "move": (row, col),
            "board": self.game.board.copy()
        })
        # advance current_history_index to the new last entry
        self.current_history_index = len(self.history) - 1

        winner = self.check_winner()
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

    @staticmethod
    def safety_net_choices(board, current_player: int, valid_moves):
        board_arr = np.array(board, copy=True)

        for move in valid_moves:
            if GameEnv.is_winning_move(board_arr, move, current_player):
                return move, valid_moves

        safe_moves = [
            move for move in valid_moves
            if not GameEnv.opponent_can_win_next(board_arr, move, current_player)
        ]

        return None, safe_moves

    @staticmethod
    def is_winning_move(board, move, player: int) -> bool:
        test_board = GameEnv._simulate_move(board, move, player)
        temp_game = TicTacToe9x9()
        temp_game.board = test_board
        return temp_game.check_winner() == player

    @staticmethod
    def opponent_can_win_next(board, move, player: int) -> bool:
        test_board = GameEnv._simulate_move(board, move, player)
        opponent = -player
        temp_game = TicTacToe9x9()
        temp_game.board = test_board
        temp_game.last_move = move
        opponent_moves = temp_game.get_valid_moves()
        for opp_move in opponent_moves:
            if GameEnv.is_winning_move(test_board, opp_move, opponent):
                return True
        return False

    @staticmethod
    def _simulate_move(board, move, player: int):
        simulated = np.array(board, copy=True)
        r, c = move
        simulated[r, c] = player
        return simulated
    
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
            data.append({"player": entry.get("player"), "move": [mv[0], mv[1]]})

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