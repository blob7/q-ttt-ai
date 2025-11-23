from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
import random
import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from game.board import PlayerPiece
from game.environment import GameEnv


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
    ):
        self.role = role
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self._episode_transitions = []  # stack
        self._table_lock = table_lock
        default_q = q_values if q_values is not None else defaultdict(_state_q_factory)
        default_visits = visit_counts if visit_counts is not None else defaultdict(_state_visit_factory)
        self.configure_tables(default_q, default_visits, table_lock=self._table_lock)

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name/identifier of the agent."""
        pass

    # --- Action selection (epsilon-greedy) ---
    def choose_action(self, env: GameEnv, learn: bool = True):
        """Select action using epsilon-greedy strategy with lazy Q-value initialization."""
        state_hash = env.get_state_hash()
        with self._lock_context():
            state_q_snapshot = self._get_state_q_snapshot(state_hash)
        verbose = False  # Set to True to see action selection details
        if verbose:
            print('-'*20)
            print(f"Choosing action for state: {state_hash} | turn {env.game.turn_count}")
            print(f"Q-values: {state_q_snapshot}")
        valid_moves = env.get_valid_moves()

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
                    q = state_q_snapshot.get(move, 0.0)
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

        # Convert defaultdicts to plain dicts for pickling
        q_values_dict = {k: dict(v) for k, v in self.q_values.items()}

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

    def _lock_context(self):
        return self._table_lock if self._table_lock is not None else nullcontext()

    def _get_state_q_snapshot(self, state):
        if isinstance(self.q_values, defaultdict):
            return dict(self.q_values[state])
        return dict(self.q_values.get(state, {}))

    def _increment_q_value(self, state, action, delta):
        if isinstance(self.q_values, defaultdict):
            state_q = self.q_values[state]
            state_q[action] = round(state_q.get(action, 0.0) + delta, 3)
        else:
            state_q = dict(self.q_values.get(state, {}))
            state_q[action] = round(state_q.get(action, 0.0) + delta, 3)
            self.q_values[state] = state_q

    def _increment_visit_count(self, state, action, increment=1):
        if isinstance(self.visit_counts, defaultdict):
            state_visits = self.visit_counts[state]
            state_visits[action] = state_visits.get(action, 0) + increment
        else:
            state_visits = dict(self.visit_counts.get(state, {}))
            state_visits[action] = state_visits.get(action, 0) + increment
            self.visit_counts[state] = state_visits

    def _merge_q_value(self, state, action, value, added_visits):
        if added_visits == 0:
            return
        if isinstance(self.q_values, defaultdict):
            state_q = self.q_values[state]
            state_visits = self.visit_counts[state]
            current_visits = state_visits.get(action, 0)
            total_visits = current_visits + added_visits
            if total_visits == 0:
                return
            prev = state_q.get(action, 0.0)
            state_q[action] = (prev * current_visits + value * added_visits) / total_visits
            state_visits[action] = total_visits
        else:
            state_q = dict(self.q_values.get(state, {}))
            state_visits = dict(self.visit_counts.get(state, {}))
            current_visits = state_visits.get(action, 0)
            total_visits = current_visits + added_visits
            if total_visits == 0:
                self.q_values[state] = state_q
                self.visit_counts[state] = state_visits
                return
            prev = state_q.get(action, 0.0)
            state_q[action] = (prev * current_visits + value * added_visits) / total_visits
            state_visits[action] = total_visits
            self.q_values[state] = state_q
            self.visit_counts[state] = state_visits

    def materialize_tables(self) -> None:
        new_q = defaultdict(_state_q_factory)
        for state, actions in self.q_values.items():
            state_q = new_q[state]
            state_q.update(dict(actions))

        new_visits = defaultdict(_state_visit_factory)
        for state, actions in self.visit_counts.items():
            state_visits = new_visits[state]
            state_visits.update(dict(actions))

        self.configure_tables(
            new_q,
            new_visits,
            table_lock=None,
        )

    def _set_state_q_values(self, state, actions: dict) -> None:
        if isinstance(self.q_values, defaultdict):
            self.q_values[state].update(actions)
        else:
            self.q_values[state] = dict(actions)

    def _set_state_visit_values(self, state, actions: dict) -> None:
        if isinstance(self.visit_counts, defaultdict):
            self.visit_counts[state].update(actions)
        else:
            self.visit_counts[state] = dict(actions)
