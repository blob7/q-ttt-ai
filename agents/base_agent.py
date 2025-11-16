from abc import ABC, abstractmethod
from collections import defaultdict
import random
import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from game.board import PlayerPiece
from game.environment import GameEnv

class BaseAgent(ABC):   
    def __init__(self, role = None, learning_rate: float = 0.01, discount_factor: float = 0.9, epsilon: float = 1.0, min_epsilon: float = 0.05, epsilon_decay: float = 0.995):
        self.role = role
        self.q_values = defaultdict(lambda: defaultdict(float))  # Q-table
        self.visit_counts = defaultdict(lambda: defaultdict(int))  # For weighted merging
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self._episode_transitions = []  # stack

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name/identifier of the agent."""
        pass

    # --- Action selection (epsilon-greedy) ---
    def choose_action(self, env: GameEnv, learn: bool = True):
        """Select action using epsilon-greedy strategy with lazy Q-value initialization."""
        state_q = self.q_values.setdefault(env.get_state_hash(), defaultdict(float))  # ensure dict exists
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
                    q = state_q.get(move, 0.0)  # lazy read
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
    def compute_reward(self, state, action, winner: Optional[int], mover) -> float:
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

            reward = self.compute_reward(state, action, winner, mover=mover)
            self.q_values[state][action] += reward
            self.visit_counts[state][action] += 1


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
                "gamma": self.gamma
            }, f)
        print(f"Agent saved to {path}")


    @classmethod
    def load(cls, file_path: str, role=None):
        path = Path(file_path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(role=role)  # Create instance
        # Wrap back into defaultdicts
        agent.q_values = defaultdict(lambda: defaultdict(float),
                                    {tuple(k): defaultdict(float, v) for k, v in data["q_values"].items()})
        agent.epsilon = data.get("epsilon", 0.2)
        agent.lr = data.get("lr", 0.1)
        agent.gamma = data.get("gamma", 0.9)
        return agent
    
    def merge_q_tables(self, qtables, visit_tables):
        """Weighted merge using visit counts."""
        for qtable, visits in zip(qtables, visit_tables):
            for state, actions in qtable.items():
                state_q = self.q_values.setdefault(state, defaultdict(float))
                state_visits = self.visit_counts.setdefault(state, defaultdict(int))
                for action, value in actions.items():
                    total_visits = state_visits[action] + visits[state][action]
                    if total_visits == 0:
                        continue
                    # Weighted average
                    state_q[action] = (
                        state_q[action] * state_visits[action] + value * visits[state][action]
                    ) / total_visits
                    state_visits[action] = total_visits
