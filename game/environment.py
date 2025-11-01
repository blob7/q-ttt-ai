# game/environment.py
from .board import TicTacToe9x9
from .utils import print_board


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
        # move for the current player (e.g. bots). Keys are 1 and -1.
        self.controllers = {1: None, -1: None}

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

    def step(self, action: tuple[int, int]):
        """Perform one move, record it, and return (state, reward, done, winner)."""
        row, col = action
        valid = self.game.make_move(row, col)

        if not valid:
            # Invalid move: penalize
            return self.get_state(), -1, True, None

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

    def register_controller(self, player: int, controller):
        """Register a controller callable for a player (1 or -1).

        The controller should be a callable taking one argument (the env)
        and returning a move tuple (row, col). Use None to unregister.
        """
        if player not in (1, -1):
            raise ValueError("player must be 1 or -1")
        self.controllers[player] = controller

    def request_current_move(self):
        """If a controller is registered for the current player, call it to
        get a move and apply it via step(). Returns the (state, reward,
        done, winner) tuple returned by step() or None if no controller.
        """
        ctrl = self.controllers.get(self.current_player)
        # Debug: surface which controller is being asked
        try:
            print(f"[Env] request_current_move: current_player={self.current_player}, controller={ctrl}")
        except Exception:
            pass
        if not ctrl:
            try:
                print("[Env] no controller registered for current player")
            except Exception:
                pass
            return None
        move = None
        try:
            move = ctrl(self)
        except Exception as e:
            # Print and re-raise to make debugging visible during runs
            try:
                print(f"[Env] controller raised exception: {e}")
            except Exception:
                pass
            raise
        # Controller may return None to indicate "no move available yet"
        # (e.g. human controller waiting for user input). In that case we
        # must not call step() — return None so callers know to wait.
        if move is None:
            try:
                print("[Env] controller returned None (waiting/human)")
            except Exception:
                pass
            return None
        try:
            print(f"[Env] controller returned move={move}")
        except Exception:
            pass
        return self.step(move)

    def render(self):
        """Prints the current board for debugging."""
        print_board(self.game.board)

    def check_winner(self, return_cells=False):
        return self.game.check_winner(return_cells=return_cells)
    
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

        # Reset board to initial state for viewing; keep history populated
        try:
            env.game.board = env.game.board * 0
            env.game.current_player = 1
            env.game.last_move = None
            env.current_history_index = -1
        except Exception:
            pass

        return env