import tkinter as tk
from typing import Callable, Optional, Tuple, Dict, List
from game.board import PlayerPiece
from gui.enums import GameMode
from gui.components import SpeedInput
from gui.runner import GameLoop
from game.environment import GameEnv
from .drawer import BoardDrawer
# Bot controller type used by the environment: callable taking GameEnv and
# returning an optional (row, col) move.
BotType = Callable[[GameEnv], Optional[Tuple[int, int]]]

class GameController:
    """Wires a View (TicTacToeGUI) to a GameEnv and registered controllers.
    Responsibilities:
    - Register bot/human controller callables in the GameEnv
    - Wire view events to handlers that mutate the GameEnv
    - Request the view refresh itself after env changes
    The controller must NOT reach into view widget internals (canvas/labels)
    — it uses the public view and drawer helper methods instead.
    """
    def __init__(
        self,
        root: tk.Misc,
        env: GameEnv,
        mode: GameMode,
        bot1: Optional[BotType] = None,
        bot2: Optional[BotType] = None,
    ) -> None:
        self.root = root
        self.env = env
        self.mode = mode
        self.bot1 = bot1
        self.bot2 = bot2
        self.bot_speed: int = 500
        # Human move queues (player -> queued moves list)
        self._human_move_queues: Dict[int, List[Tuple[int, int]]] = {1: [], -1: []}
        self._human_players: set[int] = set()
        # Register controllers in the environment. If None, register a
        # human controller that pops from the corresponding human queue.
        if self.bot1 is None:
            self.env.register_controller(PlayerPiece.X.value, lambda e, _s, _sh, p=PlayerPiece.X.value: self._pop_human_move(p))
            self._human_players.add(PlayerPiece.X.value)
        else:
            self.env.register_controller(PlayerPiece.X.value, self.bot1)
        if self.bot2 is None:
            self.env.register_controller(PlayerPiece.O.value, lambda e, _s, _sh, p=PlayerPiece.O.value: self._pop_human_move(p))
            self._human_players.add(PlayerPiece.O.value)
        else:
            self.env.register_controller(PlayerPiece.O.value, self.bot2)
        # Will be set in attach_view
        self.view = None
        self.loop: Optional[GameLoop] = None
        self.speed_input: Optional[SpeedInput] = None
        # Playback flag: True when autoplay/play is active
        self._playing = False


    def attach_view(self, view) -> None:
        """Attach the TicTacToeGUI view and wire callbacks.
        The view is responsible for layout and rendering. The controller only
        wires events and directs the view to refresh when the env changes.
        """
        self.view = view
        self.speed_input = view.speed_input
        # Prefer the drawer-level cell binding (gives (row,col) directly).
        # Fall back to the raw click binder if the drawer does not implement it.
        view.drawer.bind_cell_click(self.on_cell_click)

        view.move_panel.on_select_move = self.jump_to_move
        view.refresh_history()
        # Create GameLoop with view callbacks (view owns refresh logic)
        self.loop = GameLoop(
            self.root,
            self.env,
            self.mode,
            self.bot_speed,
            draw_cb=view.refresh_board,
            status_cb=view.refresh_status,
            history_cb=view.refresh_history,
        )
        # Wire buttons
        view.reset_button.config(command=self.reset)
        view.quit_button.config(command=self.root.destroy)
        if self.speed_input is not None:
            self.speed_input.on_change = self.update_speed
        if getattr(view, "play_button", None) is not None:
            # Wire play/pause callbacks. The PlayPauseButton will call
            # on_play/on_pause which we set to the controller methods.
            try:
                # PlayPauseButton expects two callbacks: on_play and on_pause
                view.play_button.on_play = self._on_play_pressed
                view.play_button.on_pause = self._on_pause_pressed
                view.play_button.set_playing(False)
            except Exception:
                # Fallback for older PlayButton
                view.play_button.config(command=self.start_bot_vs_bot)

        # Default playing state depends on mode:
        # - PLAYER_V_PLAYER: allow immediate play (no play button)
        # - PLAYER_V_BOT and BOT_V_BOT: start paused until the user presses Play
        try:
            if self.mode == GameMode.PLAYER_V_PLAYER:
                self._playing = True
            else:
                self._playing = False
        except Exception:
            self._playing = False

        # Ensure play button (if present) reflects initial state
        try:
            if getattr(self.view, "play_button", None) is not None:
                try:
                    self.view.play_button.set_playing(self._playing)
                except Exception:
                    pass
        except Exception:
            pass

        # Wire save/load/close match buttons
        try:
            if getattr(self.view, "save_button", None) is not None:
                self.view.save_button.config(command=self._on_save_button)
            if getattr(self.view, "load_button", None) is not None:
                self.view.load_button.config(command=self._on_load_button)
        except Exception:
            pass

        # If the current player at startup has a controller (bot), schedule
        # NOTE: Do not auto-start autoplay here. Autoplay is explicit via the
        # play button. We intentionally avoid scheduling an initial bot move
        # on attach so human-vs-bot setups don't auto-play. Immediate bot
        # replies after a human move are handled via process_available_moves.


    def update_speed(self, val: int) -> None:
        self.bot_speed = int(val)
        if self.loop is not None:
            self.loop.bot_speed = self.bot_speed


    # --- Play / Pause control ---
    def _on_play_pressed(self) -> None:
        """Callback when Play is pressed on the UI control."""
        self.play()

    def _on_pause_pressed(self) -> None:
        """Callback when Pause is pressed on the UI control."""
        self.pause()

    def play(self) -> None:
        """Start playback/autoplay. For BOT_V_BOT this is continuous; for
        PLAYER_V_BOT this starts scheduling bot responses when it's the
        bot's turn (human turns still wait for clicks).
        """
        if self._playing:
            return
        self._playing = True
        if self.view is not None and getattr(self.view, "play_button", None) is not None:
            try:
                self.view.play_button.set_playing(True)
            except Exception:
                pass

        if self.loop is None:
            return

        if self.mode == GameMode.BOT_V_BOT:
            # Continuous autoplay
            self.loop.start(autoplay=True)
        elif self.mode == GameMode.PLAYER_V_BOT:
            # Schedule a single bot step; process_available_moves will
            # chain schedule_single calls while the bot has moves.
            self.loop.schedule_single(self.bot_speed)
        elif self.mode == GameMode.VIEW_MATCH:
            # Start viewing playback: advance through recorded moves.
            self.loop.start(autoplay=True)

    def stop_view_match(self) -> None:
        """Exit view-match mode and restore the previous environment."""
        if not getattr(self, "_viewing_match", False):
            return
        try:
            # Restore env references
            old = getattr(self, "_saved_env", None)
            if old is not None:
                self.env = old
                if self.view is not None:
                    try:
                        self.view.env = old
                    except Exception:
                        pass
                    try:
                        self.view.drawer.env = old
                    except Exception:
                        pass
                if self.loop is not None:
                    self.loop.env = old
        except Exception:
            pass
        self._viewing_match = False
        # Refresh UI to reflect restored env
        if self.view is not None:
            self.view.refresh_ui()

    def pause(self) -> None:
        """Stop playback/autoplay. Cancels any scheduled callbacks."""
        if not self._playing:
            return
        self._playing = False
        if self.loop is not None:
            self.loop.stop()
        if self.view is not None and getattr(self.view, "play_button", None) is not None:
            try:
                self.view.play_button.set_playing(False)
            except Exception:
                pass

    # UI helper glue: file dialogs + delegating IO to env
    def _on_save_button(self) -> None:
        try:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if not path:
                return
            # Delegate actual export to env
            try:
                self.env.export_history(path)
            except Exception as e:
                print(f"[Controller] env.export_history failed: {e}")
        except Exception as e:
            print(f"[Controller] save dialog failed: {e}")

    def _on_load_button(self) -> None:
        try:
            from tkinter import filedialog
            path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if not path:
                return
            # Ask env to load a viewing env from file
            try:
                temp_env = GameEnv.load_from_file(path)
            except Exception as e:
                print(f"[Controller] GameEnv.load_from_file failed: {e}")
                return

            # Swap env/view to the loaded viewing env (preserve old env)
            self._saved_env = self.env
            self.env = temp_env
            if self.view is not None:
                try:
                    self.view.env = temp_env
                except Exception:
                    pass
                try:
                    self.view.drawer.env = temp_env
                except Exception:
                    pass
            # Ensure the move history panel points to the loaded env so it
            # can display the replayed moves immediately.
            try:
                if self.view is not None and getattr(self.view, "move_panel", None) is not None:
                    self.view.move_panel.env = temp_env
            except Exception:
                pass
            if self.loop is not None:
                self.loop.env = temp_env

            # Enter viewing mode and pause playback
            self._viewing_match = True
            try:
                self.pause()
            except Exception:
                pass
            if self.view is not None:
                self.view.refresh_ui()
        except Exception as e:
            print(f"[Controller] load dialog failed: {e}")

            
    def start_bot_vs_bot(self) -> None:
        if self.view is not None:
            self.view.set_play_button_state("disabled")
        if self.loop is not None:
            self.loop.start(autoplay=True)


    def _current_bot(self, env: GameEnv) -> Optional[BotType]:
        if env.current_player == 1 and self.bot1:
            return self.bot1
        if env.current_player == -1 and self.bot2:
            return self.bot2
        return None
    

    def on_cell_click(self, row: int, col: int) -> None:
        """Handle a click expressed as a (row, col) cell coordinate."""
        # Ignore clicks while paused — clicking should not mutate turn
        # order when playback is paused.
        if not self._playing:
            return
        if self.mode == GameMode.BOT_V_BOT or self.env.check_winner():
            return
        if (row, col) not in self.env.get_valid_moves():
            return
        current = self.env.current_player
        if current not in self._human_players:
            return
        self._human_move_queues[current].append((row, col))
        # Only trigger bot processing if playback is active.
        if self.loop is not None and self._playing:
            self.loop.process_available_moves()


    def reset(self) -> None:
        self.env.reset()
        if self.view is not None:
            # For modes with autoplay (bots) pause on reset; keep PvP
            # interactive without requiring Play.
            try:
                if self.mode != GameMode.PLAYER_V_PLAYER:
                    self.pause()
            except Exception:
                pass
            # Clear any queued human moves to avoid out-of-order execution
            try:
                self._human_move_queues = {1: [], -1: []}
            except Exception:
                pass
            self.view.refresh_ui(set_play_state="normal")

            
    def jump_to_move(self, index: int) -> None:
        self.env.jump_to_move(index)
        if self.view is not None:
            # Jumping to history should pause playback for bot modes; keep
            # PvP interactive.
            try:
                if self.mode != GameMode.PLAYER_V_PLAYER:
                    self.pause()
            except Exception:
                pass
            try:
                # Clear queued moves when jumping in history
                self._human_move_queues = {1: [], -1: []}
            except Exception:
                pass
            self.view.refresh_ui()

                        
    def make_move(self, move_fn: Optional[BotType] = None, autoplay: bool = False) -> None:
        if move_fn:
            move = move_fn(self.env)
            if move is None:
                return
            self.env.step(move)
        if self.view is not None:
            self.view.refresh_ui()
        if self.env.check_winner():
            if self.mode == GameMode.BOT_V_BOT and self.view is not None:
                self.view.set_play_button_state("normal")
            return
        if autoplay and self.mode == GameMode.BOT_V_BOT and self.loop is not None:
            self.loop.start(autoplay=True)


    def _env_bot_play(self, autoplay: bool = False) -> None:
        if self.env.check_winner():
            return
        # Delegate to the GameLoop step_once which now handles controller
        # invocation and env.step(). This keeps orchestration out of GameEnv.
        if self.loop is None:
            return
        done = self.loop.step_once()
        if self.view is not None:
            self.view.refresh_ui()
        if done:
            if self.mode == GameMode.BOT_V_BOT and self.view is not None:
                self.view.set_play_button_state("normal")
            return
        if autoplay and self.mode == GameMode.BOT_V_BOT and self.loop is not None:
            self.loop.schedule_single(self.bot_speed)


    def _pop_human_move(self, player: int) -> Optional[Tuple[int, int]]:
        q = self._human_move_queues.get(player)
        if not q:
            return None
        return q.pop(0)