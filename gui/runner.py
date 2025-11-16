from typing import Optional
from game.environment import GameEnv
from gui.enums import GameMode


class GameLoop:
    """Orchestrates asking controllers for moves, applying them via the
    environment, updating the UI and scheduling subsequent moves using a
    Tk root's `after` method.

    This keeps scheduling and continuation logic out of the GUI view code.
    """
    def __init__(self, root, env: GameEnv, mode, bot_speed: int, draw_cb, status_cb, history_cb):
        self.root = root
        self.env = env
        self.mode = mode
        self.bot_speed = bot_speed
        self.draw_cb = draw_cb
        self.status_cb = status_cb
        self.history_cb = history_cb

        self._after_id = None
        self._running = False

    def start(self, autoplay: bool = False):
        """Start the loop. If autoplay=True and there is a controller for the
        current player the loop will continue automatically (used for
        Bot-vs-Bot)."""
        if self._running:
            return
        self._running = True
        # For a start we schedule a single step after bot_speed so UI has a
        # small delay before bots begin.
        self._schedule(self.bot_speed, autoplay)

    def stop(self):
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = None
        self._running = False

    def step_once(self):
        """Perform a single controller request -> env.step cycle and update UI.
        Returns True if the game ended as a result of the applied move.
        """
        # VIEW_MATCH behaves differently: step through the recorded history
        # by incrementing the current_history_index and calling jump_to_move.
        if self.mode == GameMode.VIEW_MATCH:
            if not self.env.history:
                return False
            next_idx = self.env.current_history_index + 1
            # If already at or past the last move, nothing to do
            if next_idx >= len(self.env.history):
                return True
            # Jump to the next move and update UI
            self.env.jump_to_move(next_idx)
            try:
                self.draw_cb()
                self.status_cb()
                self.history_cb()
            except Exception:
                print("Error updating UI components after view-step")
            # Done when we've reached the last move
            done = (self.env.current_history_index == len(self.env.history) - 1)
            return bool(done)

        # Non-viewing mode: ask the registered controller for the
        # current player to produce a move, then apply it via env.step().
        ctrl = None
        try:
            ctrl = self.env.get_controller_for_current_player()
        except Exception:
            ctrl = None
        if not ctrl:
            # No controller registered (e.g. human waiting for input)
            return False
        try:
            # Controllers in this project expect (state, valid_moves)
            # for bots and (env) for simple human lambdas. Provide the
            # (state, valid_moves) form and allow controllers to return
            # a list or tuple; normalize lists/arrays to tuple before
            # passing to env.step().
            move = ctrl(self.env.get_state(), self.env.get_state_hash(), self.env.get_valid_moves())
        except Exception as e:
            try:
                print(f"[GameLoop] controller raised exception: {e}")
            except Exception:
                pass
            raise
        # Controller may return None to indicate "waiting" (human input)
        if move is None:
            return False
        # Normalize move to an immutable tuple so later code can use it as
        # a dict key (agents/Q-table) without raising 'unhashable type: list'.
        try:
            if isinstance(move, list):
                move = tuple(move)
        except Exception:
            pass
        # Apply the move via the environment
        res = self.env.step(move)
        # Update UI
        try:
            self.draw_cb()
            self.status_cb()
            self.history_cb()
        except Exception:
            print("Error updating UI components after step")
        done = res[2]
        return bool(done)

    def process_available_moves(self) -> bool:
        """Process controller moves repeatedly until a controller returns
        None (i.e. waiting human input) or the game ends. Returns True if
        the game ended during processing.
        This is used to apply immediate bot replies after a human move
        without enabling global autoplay.
        """
        # Process a single available move, update UI, and then schedule the
        # next step after `bot_speed` ms. This ensures the speed control is
        # respected for Player-v-Bot flows while avoiding a tight synchronous
        # loop that would block the UI.
        # Special-case viewing mode: step through history rather than
        # asking controllers for moves.
        if self.mode == GameMode.VIEW_MATCH:
            if not self.env.history:
                return False
            next_idx = self.env.current_history_index + 1
            if next_idx >= len(self.env.history):
                return True
            self.env.jump_to_move(next_idx)
            try:
                self.draw_cb()
            except Exception:
                pass
            try:
                self.status_cb()
            except Exception:
                pass
            try:
                self.history_cb()
            except Exception:
                pass
            done = (self.env.current_history_index == len(self.env.history) - 1)
            if done:
                return True
            self.schedule_single(self.bot_speed)
            return False

        ctrl = None
        try:
            ctrl = self.env.get_controller_for_current_player()
        except Exception:
            ctrl = None
        if not ctrl:
            return False
        try:
            move = ctrl(self.env.get_state(), self.env.get_state_hash(), self.env.get_valid_moves())
        except Exception as e:
            print(f"[GameLoop] controller raised exception: {e}")
            raise
        # Normalize move to tuple to avoid unhashable-list issues later
        try:
            if isinstance(move, list):
                move = tuple(move)
        except Exception:
            pass
        if move is None:
            return False
        res = self.env.step(move)
        # Update UI after applying the move
        try:
            self.draw_cb()
            self.status_cb()
            self.history_cb()
        except Exception:
            print("Error updating UI components after step")
        done = bool(res[2])
        if done:
            return True
        # Not finished: schedule the next controller step after bot_speed ms.
        self.schedule_single(self.bot_speed)
        return False

    def schedule_single(self, delay: Optional[int] = None):
        """Schedule one controller step after `delay` ms (default bot_speed)."""
        if delay is None:
            delay = self.bot_speed
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = self.root.after(delay, lambda: self._run_step(autoplay=False))

    def _schedule(self, delay: int, autoplay: bool):
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = self.root.after(delay, lambda: self._run_step(autoplay=autoplay))

    def _run_step(self, autoplay: bool):
        self._after_id = None
        # If not running, allow single scheduled steps (autoplay=False)
        # to execute — this is used to schedule a single bot response in
        # Player-vs-Bot mode. Only block when autoplay=True and the loop
        # isn't running.
        if not self._running and autoplay:
            return

        ended = self.step_once()
        if ended:
            self.stop()
            return

        # Continue when autoplay is True for modes that should auto-advance.
        # Bot-vs-Bot and VIEW_MATCH both require continuous scheduling.
        if autoplay and self.mode in (GameMode.BOT_V_BOT, GameMode.VIEW_MATCH):
            # schedule another step
            self._schedule(self.bot_speed, autoplay=True)
