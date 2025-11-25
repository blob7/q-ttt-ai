from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
import copy
import math
import random
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from agents.shared_backend import SharedActionValueBackend
from game.board import PlayerPiece
from game.environment import GameEnv
from game.utils import decode_move, encode_move


def _state_q_factory():
    return defaultdict(float)


def _state_visit_factory():
    return defaultdict(int)

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
        visit_counts=None,
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
        default_visits = visit_counts if visit_counts is not None else defaultdict(_state_visit_factory)
        self.configure_tables(default_q, default_visits, table_lock=self._table_lock)
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
                    q = state_q_snapshot.get(canonical_move)
                    if q is None and canonical_move != move:
                        q = state_q_snapshot.get(move)
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
                self._increment_visit_count(state, action)


    def save(self, file_path: str):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.uses_shared_backend():
            raw_q, _ = self._shared_backend.export_tables()  # type: ignore[union-attr]
            q_values_dict: Dict[tuple[bytes, int, int], Dict[tuple[int, int], float]] = {}
            for state, actions in raw_q.items():
                decoded: Dict[tuple[int, int], float] = {}
                for action_key, value in actions.items():
                    try:
                        move = self._decode_action(state, action_key)
                    except ValueError:
                        continue
                    decoded[move] = value
                if decoded:
                    q_values_dict[state] = decoded
        else:
            q_values_dict = {state: dict(actions) for state, actions in self._iter_mapping_items(self.q_values)}

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
        return agent
    
    def merge_q_tables(self, qtables, visit_tables):
        """Weighted merge using visit counts."""
        with self._lock_context():
            for qtable, visits in zip(qtables, visit_tables):
                for state, actions in qtable.items():
                    visit_source = visits.get(state, {})
                    for action, value in actions.items():
                        add_visits = visit_source.get(action, 0)
                        self._merge_q_value(state, action, value, add_visits)

    # ------------------
    # Internal helpers
    # ------------------
    def configure_tables(
        self,
        q_values,
        visit_counts,
        *,
        table_lock=None,
    ) -> None:
        self.q_values = q_values
        self.visit_counts = visit_counts
        self._table_lock = table_lock
        self._shared_backend = None
        self._shared_backend_owner = False
        self._track_deltas = False
        self._pending_q_deltas = defaultdict(_state_q_factory)
        self._pending_visit_deltas = defaultdict(_state_visit_factory)

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
                for action, value in dict(actions).items():
                    action_key = self._encode_action(state, action)
                    backend.set_q_value(state, action_key, value)
            for state, actions in self._iter_mapping_items(self.visit_counts):
                if not actions:
                    continue
                for action, count in dict(actions).items():
                    action_key = self._encode_action(state, action)
                    backend.set_visit_count(state, action_key, count)

        self._shared_backend = backend
        self._shared_backend_owner = own
        self.q_values = backend
        self.visit_counts = backend
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
        clone._pending_visit_deltas = defaultdict(_state_visit_factory)
        clone._shared_backend_owner = False
        if self.uses_shared_backend():
            backend = self._shared_backend
            clone._shared_backend = backend.fork() if backend is not None else None
            clone.q_values = clone._shared_backend  # type: ignore[assignment]
            clone.visit_counts = clone._shared_backend  # type: ignore[assignment]
        else:
            clone._shared_backend = None
            clone.q_values = self.q_values
            clone.visit_counts = self.visit_counts
        return clone

    def enable_delta_tracking(self) -> None:
        with self._lock_context():
            self._track_deltas = True
            self._pending_q_deltas = defaultdict(_state_q_factory)
            self._pending_visit_deltas = defaultdict(_state_visit_factory)

    def disable_delta_tracking(self) -> None:
        with self._lock_context():
            self._track_deltas = False
            self._pending_q_deltas = defaultdict(_state_q_factory)
            self._pending_visit_deltas = defaultdict(_state_visit_factory)

    def drain_deltas(self) -> tuple[Dict, Dict]:
        with self._lock_context():
            q_delta = {state: dict(actions) for state, actions in self._pending_q_deltas.items() if actions}
            visit_delta = {state: dict(actions) for state, actions in self._pending_visit_deltas.items() if actions}
            self._pending_q_deltas = defaultdict(_state_q_factory)
            self._pending_visit_deltas = defaultdict(_state_visit_factory)
            return q_delta, visit_delta

    def apply_deltas(self, q_delta: Dict, visit_delta: Dict) -> None:
        with self._lock_context():
            for state, actions in q_delta.items():
                for action, delta in actions.items():
                    self._increment_q_value(state, action, delta, track=False)
            for state, actions in visit_delta.items():
                for action, inc in actions.items():
                    self._increment_visit_count(state, action, inc, track=False)

    def _get_state_q_snapshot(self, state):
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return {}
            encoded = backend.get_state_q_snapshot(state)
            if not encoded:
                return {}
            decoded: Dict[tuple[int, int], float] = {}
            for action_key, value in encoded.items():
                try:
                    move = self._decode_action(state, action_key)
                except ValueError:
                    continue
                decoded[move] = value
            return decoded
        if isinstance(self.q_values, defaultdict):
            return dict(self.q_values[state])
        if isinstance(self.q_values, Mapping):
            return dict(self.q_values.get(state, {}))  # type: ignore[arg-type]
        return {}

    def _increment_q_value(self, state, action, delta, *, track: bool = True):
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            action_key = self._encode_action(state, action)
            applied = backend.increment_q_value(state, action_key, delta, rounding=3)
            if track and getattr(self, "_track_deltas", False) and applied != 0.0:
                pending = self._pending_q_deltas[state]
                pending[action] = pending.get(action, 0.0) + applied
                if pending[action] == 0.0:
                    pending.pop(action, None)
                if not pending:
                    self._pending_q_deltas.pop(state, None)
            return

        if isinstance(self.q_values, defaultdict):
            state_q = self.q_values[state]
        elif isinstance(self.q_values, Mapping):
            state_q = dict(self.q_values.get(state, {}))  # type: ignore[arg-type]
            self.q_values[state] = state_q  # type: ignore[index]
        else:
            state_q = {}

        prev = state_q.get(action, 0.0)
        new_value = round(prev + delta, 3)
        state_q[action] = new_value

        if track and getattr(self, "_track_deltas", False):
            pending = self._pending_q_deltas[state]
            applied = new_value - prev
            pending[action] = pending.get(action, 0.0) + applied
            if pending[action] == 0.0:
                pending.pop(action, None)
            if not pending:
                self._pending_q_deltas.pop(state, None)

    def _increment_visit_count(self, state, action, increment=1, *, track: bool = True):
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            action_key = self._encode_action(state, action)
            applied = backend.increment_visit_count(state, action_key, increment)
            if track and getattr(self, "_track_deltas", False) and applied != 0:
                pending = self._pending_visit_deltas[state]
                pending[action] = pending.get(action, 0) + applied
                if pending[action] == 0:
                    pending.pop(action, None)
                if not pending:
                    self._pending_visit_deltas.pop(state, None)
            return

        if isinstance(self.visit_counts, defaultdict):
            state_visits = self.visit_counts[state]
        elif isinstance(self.visit_counts, Mapping):
            state_visits = dict(self.visit_counts.get(state, {}))  # type: ignore[arg-type]
            self.visit_counts[state] = state_visits  # type: ignore[index]
        else:
            state_visits = {}

        prev = state_visits.get(action, 0)
        new_count = prev + increment
        state_visits[action] = new_count

        if track and getattr(self, "_track_deltas", False):
            pending = self._pending_visit_deltas[state]
            applied = new_count - prev
            pending[action] = pending.get(action, 0) + applied
            if pending[action] == 0:
                pending.pop(action, None)
            if not pending:
                self._pending_visit_deltas.pop(state, None)

    def _merge_q_value(self, state, action, value, added_visits):
        if added_visits == 0:
            return
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            action_key = self._encode_action(state, action)
            backend.merge_q_value(state, action_key, value, added_visits)
            return
        if isinstance(self.q_values, defaultdict) and isinstance(self.visit_counts, defaultdict):
            state_q = self.q_values[state]
            state_visits = self.visit_counts[state]
            current_visits = state_visits.get(action, 0)
            total_visits = current_visits + added_visits
            if total_visits == 0:
                return
            prev = state_q.get(action, 0.0)
            state_q[action] = (prev * current_visits + value * added_visits) / total_visits
            state_visits[action] = total_visits
        elif isinstance(self.q_values, Mapping) and isinstance(self.visit_counts, Mapping):
            state_q = dict(self.q_values.get(state, {}))  # type: ignore[arg-type]
            state_visits = dict(self.visit_counts.get(state, {}))  # type: ignore[arg-type]
            current_visits = state_visits.get(action, 0)
            total_visits = current_visits + added_visits
            if total_visits == 0:
                self.q_values[state] = state_q  # type: ignore[index]
                self.visit_counts[state] = state_visits  # type: ignore[index]
                return
            prev = state_q.get(action, 0.0)
            state_q[action] = (prev * current_visits + value * added_visits) / total_visits
            state_visits[action] = total_visits
            self.q_values[state] = state_q  # type: ignore[index]
            self.visit_counts[state] = state_visits  # type: ignore[index]

    def materialize_tables(self) -> None:
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            raw_q, raw_visits = backend.export_tables()
            new_q = defaultdict(_state_q_factory)
            for state, actions in raw_q.items():
                state_q = new_q[state]
                for action_key, value in actions.items():
                    try:
                        move = self._decode_action(state, action_key)
                    except ValueError:
                        continue
                    state_q[move] = value
            new_visits = defaultdict(_state_visit_factory)
            for state, actions in raw_visits.items():
                state_visits = new_visits[state]
                for action_key, count in actions.items():
                    try:
                        move = self._decode_action(state, action_key)
                    except ValueError:
                        continue
                    state_visits[move] = count
            self.close_shared_backend(unlink=False)
            self.configure_tables(new_q, new_visits, table_lock=None)
            return

        new_q = defaultdict(_state_q_factory)
        for state, actions in self._iter_mapping_items(self.q_values):
            state_q = new_q[state]
            state_q.update(dict(actions))

        new_visits = defaultdict(_state_visit_factory)
        for state, actions in self._iter_mapping_items(self.visit_counts):
            state_visits = new_visits[state]
            state_visits.update(dict(actions))

        self.configure_tables(new_q, new_visits, table_lock=None)

    def _set_state_q_values(self, state, actions: dict) -> None:
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            for action, value in actions.items():
                action_key = self._encode_action(state, action)
                backend.set_q_value(state, action_key, value)
            return
        if isinstance(self.q_values, defaultdict):
            self.q_values[state].update(actions)
        elif isinstance(self.q_values, Mapping):
            self.q_values[state] = dict(actions)  # type: ignore[index]

    def _set_state_visit_values(self, state, actions: dict) -> None:
        if self.uses_shared_backend():
            backend = self._shared_backend  # type: ignore[assignment]
            if backend is None:
                return
            for action, count in actions.items():
                action_key = self._encode_action(state, action)
                backend.set_visit_count(state, action_key, count)
            return
        if isinstance(self.visit_counts, defaultdict):
            self.visit_counts[state].update(actions)
        elif isinstance(self.visit_counts, Mapping):
            self.visit_counts[state] = dict(actions)  # type: ignore[index]
