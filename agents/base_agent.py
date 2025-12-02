from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from contextlib import nullcontext
import copy
import math
import random
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from agents.shared_backend import SharedActionValueBackend
from game.board import PlayerPiece
from game.environment import GameEnv
from game.utils import decode_move, encode_move

import sys

def _state_q_factory():
    return defaultdict(float)


ActionLike = int | tuple[int, int]

class BaseAgent(ABC):   
    def __init__(
        self,
        role=None,
        learning_rate: float = 0.01,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        *,
        q_values=None,
        table_lock=None,
        shared_backend: Optional[SharedActionValueBackend] = None,
        shared_backend_owner: bool = False,
    ):
        self.role = role
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self._episode_transitions = []  # stack
        self._table_lock = table_lock
        self._shared_backend: Optional[SharedActionValueBackend] = None
        self._shared_backend_owner = False
        default_q = q_values if q_values is not None else defaultdict(_state_q_factory)
        self.configure_tables(default_q, table_lock=self._table_lock)
        if shared_backend is not None:
            self.use_shared_backend(shared_backend, own=shared_backend_owner)


    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name/identifier of the agent."""
        pass

    # --- Action selection (epsilon-greedy) ---
    def choose_action(self, env: GameEnv, learn: bool = True):
        """Select action using epsilon-greedy strategy with lazy Q-value initialization."""
        state_hash, move_to_canonical, _ = env.get_canonical_state()
        with self._lock_context():
            state_q_snapshot = self._get_state_q_snapshot(state_hash)
        verbose = False  # Set to True to see action selection details
        if verbose:
            print('-'*20)
            print(f"Choosing action for state: {state_hash} | turn {env.game.turn_count}")
            print(f"Q-values: {state_q_snapshot}")
        valid_moves = env.get_valid_moves()
        canonical_for_move: Dict[tuple[int, int], tuple[int, int]] = {
            move: move_to_canonical(move)
            for move in valid_moves
        }
        canonical_action_keys: Dict[tuple[int, int], int] = {
            move: self._action_key(state_hash, canonical_move)
            for move, canonical_move in canonical_for_move.items()
        }
        fallback_action_keys: Dict[tuple[int, int], int] = {
            move: self._action_key(state_hash, move)
            for move, canonical_move in canonical_for_move.items()
            if canonical_move != move
        }

        safety_move, safe_moves = env.safety_net_choices()
        selected_move = None

        if safety_move is not None:
            selected_move = safety_move
        else:
            eligible_moves = safe_moves if safe_moves else valid_moves

            # Epsilon-greedy policy
            if learn and random.random() < self.epsilon:
                # Explore: random valid move
                selected_move = random.choice(eligible_moves)
            else:
                # Exploit: pick move with max Q-value (default to 0.0 if unseen)
                best_move = None
                best_value = float('-inf')
                for move in eligible_moves:
                    canonical_move = canonical_for_move[move]
                    action_key = canonical_action_keys[move]
                    q = state_q_snapshot.get(action_key)
                    if q is None:
                        fallback_key = fallback_action_keys.get(move)
                        if fallback_key is not None:
                            q = state_q_snapshot.get(fallback_key)
                    if q is None:
                        q = 0.0
                    if q > best_value:
                        best_move = move
                        best_value = q
                selected_move = best_move if best_move is not None else random.choice(valid_moves)

        if selected_move is None and valid_moves:
            selected_move = random.choice(valid_moves)

        return selected_move
                
    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    # --- Reward computation ---
    @abstractmethod
    def compute_reward(self, state, action, winner: Optional[int], mover, steps_from_end: int) -> float:
        """Must be implemented by each agent.
        Determines the reward for the last action based on agent's own logic."""
        pass


    def learn_result(
        self,
        winner: Optional[int],
        state_history: list[dict],
        learn_from: int | tuple[int, int] = (PlayerPiece.X.value, PlayerPiece.O.value)
    ) -> None:
        # Normalize input to a set
        if isinstance(learn_from, int):
            learn_set = {learn_from}
        else:
            learn_set = set(learn_from)

        # state_history is chronological: index 0 = first move
        total_moves = len(state_history)

        # Reverse order for reward discounting
        with self._lock_context():
            for reverse_idx, transition in enumerate(reversed(state_history)):
                real_idx = total_moves - 1 - reverse_idx

                # Determine which player made this move
                if real_idx % 2 == 0:
                    mover = PlayerPiece.X.value
                else:
                    mover = PlayerPiece.O.value

                # Skip if this move is not from the desired player(s)
                if mover not in learn_set:
                    continue

                state = transition["state"]
                action = transition["action"]

                reward = self.compute_reward(state, action, winner, mover, reverse_idx)
                self._increment_q_value(state, action, reward)


    def save(self, file_path: str):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.uses_shared_backend():
            raw_q = self._shared_backend.export_tables()  # type: ignore[union-attr]
            q_values_dict: Dict[tuple[bytes, int, int], Dict[int, float]] = {
                state: {int(action_key): float(value) for action_key, value in actions.items()}
                for state, actions in raw_q.items()
                if actions
            }
        else:
            q_values_dict: Dict[tuple[bytes, int, int], Dict[int, float]] = {}
            for state, actions in self._iter_mapping_items(self.q_values):
                snapshot = self._snapshot_action_values(state, actions, mutate=False)
                if snapshot:
                    q_values_dict[state] = snapshot

        with open(path, "wb") as f:
            pickle.dump({
                "q_values": q_values_dict,
                "epsilon": self.epsilon,
                "lr": self.lr,
                "discount_factor": self.discount_factor
            }, f)
        print(f"Agent saved to {path}")


    @classmethod
    def load(cls, file_path: str, role=None):
        path = Path(file_path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(role=role)  # Create instance
        for state, actions in data.get("q_values", {}).items():
            agent._set_state_q_values(state, actions)
        agent.epsilon = data.get("epsilon", 0.2)
        agent.lr = data.get("lr", 0.1)
        agent.discount_factor = data.get("discount_factor", 0.9)

        print(f"Agent loaded from {path}")
        print(f'Q-Value Size after load: ', sys.getsizeof(agent.q_values), 'Bytes')
        return agent
    
    def merge_q_tables(self, qtables) -> None:
        """Merge external Q-tables by overwriting with the latest values."""
        with self._lock_context():
            for qtable in qtables:
                for state, actions in qtable.items():
                    for action, value in actions.items():
                        self._merge_q_value(state, action, value)

    # ------------------
    # Internal helpers
    # ------------------
    def configure_tables(
        self,
        q_values,
        *,
        table_lock=None,
    ) -> None:
        self.q_values = q_values
        self._table_lock = table_lock
        self._shared_backend = None
        self._shared_backend_owner = False
        self._track_deltas = False
        self._pending_q_deltas = defaultdict(_state_q_factory)

    def _lock_context(self):
        return self._table_lock if self._table_lock is not None else nullcontext()

    def uses_shared_backend(self) -> bool:
        return self._shared_backend is not None

    def get_shared_backend(self) -> Optional[SharedActionValueBackend]:
        return self._shared_backend

    @staticmethod
    def _state_board_size(state: tuple[bytes, int, int]) -> int:
        board_bytes = state[0]
        cells = len(board_bytes)
        if cells <= 0:
            return 0
        size = math.isqrt(cells)
        if size * size != cells:
            raise ValueError(f"State board bytes length {cells} is not a perfect square")
        return size

    def _encode_action(self, state: tuple[bytes, int, int], action: tuple[int, int]) -> int:
        board_size = self._state_board_size(state)
        return encode_move(action, board_size)

    def _decode_action(self, state: tuple[bytes, int, int], action_key: int) -> tuple[int, int]:
        board_size = self._state_board_size(state)
        move = decode_move(action_key, board_size)
        if move is None:
            raise ValueError("Encountered invalid action key for decoding")
        return move

    def _action_key(self, state: tuple[bytes, int, int], action: ActionLike | None) -> int:
        if isinstance(action, int):
            return action
        if action is None:
            raise ValueError("Action cannot be None")
        return self._encode_action(state, action)

    def _ensure_int_action_keys(self, state: tuple[bytes, int, int], actions: MutableMapping) -> None:
        if not actions:
            return
        removals: list = []
        updates: list[tuple[int, object]] = []
        for raw_key, value in list(actions.items()):
            if isinstance(raw_key, int):
                continue
            try:
                action_key = self._action_key(state, raw_key)
            except ValueError:
                removals.append(raw_key)
                continue
            removals.append(raw_key)
            updates.append((action_key, value))
        for raw_key in removals:
            actions.pop(raw_key, None)
        for action_key, value in updates:
            actions[action_key] = value

    def _snapshot_values(self, state: tuple[bytes, int, int], actions, *, mutate: bool, cast_fn: Callable[[object], object]) -> Dict[int, object]:
        if mutate and isinstance(actions, MutableMapping):
            self._ensure_int_action_keys(state, actions)
            source = actions.items()
        else:
            source = actions.items()

        snapshot: Dict[int, object] = {}
        for raw_key, value in source:
            if isinstance(raw_key, int):
                key = int(raw_key)
            else:
                try:
                    key = self._action_key(state, raw_key)
                except ValueError:
                    continue
            snapshot[key] = cast_fn(value)
        return snapshot

    def _snapshot_action_values(self, state: tuple[bytes, int, int], actions, *, mutate: bool) -> Dict[int, float]:
        cast: Callable[[object], float] = lambda v: float(v)
        raw_snapshot = self._snapshot_values(state, actions, mutate=mutate, cast_fn=cast)
        return {key: float(value) for key, value in raw_snapshot.items()}

    @staticmethod
    def _iter_mapping_items(table) -> list[tuple]:
        if isinstance(table, Mapping):
            return list(table.items())
        return []

    def use_shared_backend(
        self,
        backend: SharedActionValueBackend,
        *,
        own: bool = False,
        migrate: bool = True,
    ) -> None:
        if backend is None:
            raise ValueError("Backend instance required")
        if self.uses_shared_backend():
            if self._shared_backend is backend:
                self._shared_backend_owner = self._shared_backend_owner or own
                return
            raise RuntimeError("Agent already bound to a different shared backend")

        if migrate:
            for state, actions in self._iter_mapping_items(self.q_values):
                if not actions:
                    continue
                snapshot = self._snapshot_action_values(state, actions, mutate=False)
                for action_key, value in snapshot.items():
                    backend.set_q_value(state, action_key, value)

        self._shared_backend = backend
        self._shared_backend_owner = own
        self.q_values = backend
        self.disable_delta_tracking()

    def close_shared_backend(self, *, unlink: bool = False) -> None:
        if self._shared_backend is None:
            return
        if self._shared_backend_owner:
            self._shared_backend.close(unlink=unlink)
        self._shared_backend = None
        self._shared_backend_owner = False

    def fork_shared(self) -> "BaseAgent":
        clone = copy.copy(self)
        clone._episode_transitions = []
        clone._track_deltas = False
        clone._pending_q_deltas = defaultdict(_state_q_factory)
        clone._shared_backend_owner = False
        if self.uses_shared_backend():
            backend = self._shared_backend
            clone._shared_backend = backend.fork() if backend is not None else None
            clone.q_values = clone._shared_backend  # type: ignore[assignment]
        else:
            clone._shared_backend = None
            clone.q_values = self.q_values
        return clone

    def enable_delta_tracking(self) -> None:
        with self._lock_context():
            self._track_deltas = True
            self._pending_q_deltas = defaultdict(_state_q_factory)

    def disable_delta_tracking(self) -> None:
        with self._lock_context():
            self._track_deltas = False
            self._pending_q_deltas = defaultdict(_state_q_factory)

    def drain_deltas(self) -> Dict:
        with self._lock_context():
            q_delta = {
                state: self._snapshot_action_values(state, actions, mutate=True)
                for state, actions in self._pending_q_deltas.items()
                if actions
            }
            self._pending_q_deltas = defaultdict(_state_q_factory)
            return q_delta

    def apply_deltas(self, q_delta: Dict) -> None:
        with self._lock_context():
            for state, actions in q_delta.items():
                for action, delta in actions.items():
                    self._increment_q_value(state, action, delta, track=False)

    def _get_state_q_snapshot(self, state):
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return {}
            encoded = backend.get_state_q_snapshot(state)
            return dict(encoded)
        if isinstance(self.q_values, defaultdict):
            state_actions = self.q_values[state]
            if not state_actions:
                return {}
            return self._snapshot_action_values(state, state_actions, mutate=True)
        if isinstance(self.q_values, MutableMapping):
            state_actions = self.q_values.get(state)
            if state_actions is None:
                state_actions = _state_q_factory()
                self.q_values[state] = state_actions
            return self._snapshot_action_values(state, state_actions, mutate=True)
        if isinstance(self.q_values, Mapping):
            actions = self.q_values.get(state, {})  # type: ignore[arg-type]
            if not actions:
                return {}
            return self._snapshot_action_values(state, dict(actions), mutate=False)
        return {}

    def _increment_q_value(self, state, action, delta, *, track: bool = True):
        action_key = self._action_key(state, action)
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            applied = backend.increment_q_value(state, action_key, delta, rounding=3)
            if track and getattr(self, "_track_deltas", False) and applied != 0.0:
                pending = self._pending_q_deltas[state]
                if pending:
                    self._ensure_int_action_keys(state, pending)
                pending[action_key] = pending.get(action_key, 0.0) + applied
                if pending[action_key] == 0.0:
                    pending.pop(action_key, None)
                if not pending:
                    self._pending_q_deltas.pop(state, None)
            return

        replace_state_q = False
        if isinstance(self.q_values, defaultdict):
            state_q = self.q_values[state]
        elif isinstance(self.q_values, MutableMapping):
            state_q = self.q_values.get(state)
            if state_q is None:
                state_q = _state_q_factory()
            replace_state_q = True
        elif isinstance(self.q_values, Mapping):
            state_q = dict(self.q_values.get(state, {}))  # type: ignore[arg-type]
        else:
            state_q = _state_q_factory()
            replace_state_q = isinstance(self.q_values, MutableMapping)

        if isinstance(state_q, MutableMapping):
            self._ensure_int_action_keys(state, state_q)
        else:
            state_q = dict(state_q)
            self._ensure_int_action_keys(state, state_q)
            replace_state_q = replace_state_q or isinstance(self.q_values, MutableMapping)

        prev = state_q.get(action_key, 0.0)
        new_value = round(prev + delta, 3)
        state_q[action_key] = new_value

        if replace_state_q and isinstance(self.q_values, MutableMapping):
            self.q_values[state] = state_q  # type: ignore[index]

        if track and getattr(self, "_track_deltas", False):
            pending = self._pending_q_deltas[state]
            if pending:
                self._ensure_int_action_keys(state, pending)
            applied = new_value - prev
            if applied != 0.0:
                pending[action_key] = pending.get(action_key, 0.0) + applied
                if pending[action_key] == 0.0:
                    pending.pop(action_key, None)
                if not pending:
                    self._pending_q_deltas.pop(state, None)


    def _merge_q_value(self, state, action, value):
        action_key = self._action_key(state, action)
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            backend.set_q_value(state, action_key, value)
            return
        if isinstance(self.q_values, defaultdict):
            target_table = self.q_values[state]
        elif isinstance(self.q_values, MutableMapping):
            target_table = self.q_values.get(state)
            if target_table is None:
                target_table = {}
                self.q_values[state] = target_table  # type: ignore[index]
        else:
            return

        if isinstance(target_table, MutableMapping):
            self._ensure_int_action_keys(state, target_table)
            target_table[action_key] = float(value)
        else:
            new_table = dict(target_table)
            self._ensure_int_action_keys(state, new_table)
            new_table[action_key] = float(value)
            if isinstance(self.q_values, MutableMapping):
                self.q_values[state] = new_table  # type: ignore[index]

    def materialize_tables(self) -> None:
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            raw_q = backend.export_tables()
            new_q = defaultdict(_state_q_factory)
            for state, actions in raw_q.items():
                state_q = new_q[state]
                for action_key, value in actions.items():
                    state_q[int(action_key)] = float(value)
            self.close_shared_backend(unlink=False)
            self.configure_tables(new_q, table_lock=None)
            return

        new_q = defaultdict(_state_q_factory)
        for state, actions in self._iter_mapping_items(self.q_values):
            snapshot = self._snapshot_action_values(state, actions, mutate=False)
            if snapshot:
                new_q[state].update(snapshot)

        self.configure_tables(new_q, table_lock=None)

    def _set_state_q_values(self, state, actions: dict) -> None:
        normalized = self._snapshot_action_values(state, actions, mutate=False)
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            for action_key, value in normalized.items():
                backend.set_q_value(state, action_key, value)
            return
        if isinstance(self.q_values, defaultdict):
            self.q_values[state].update(normalized)
        elif isinstance(self.q_values, MutableMapping):
            self.q_values[state] = dict(normalized)  # type: ignore[index]
        elif isinstance(self.q_values, Mapping):
            self.q_values[state] = dict(normalized)  # type: ignore[index]

