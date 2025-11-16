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
    def choose_action(self, state, state_hash, valid_moves, learn: bool = True):
        """Select action using epsilon-greedy strategy with lazy Q-value initialization."""
        state_q = self.q_values.setdefault(state_hash, defaultdict(float))  # ensure dict exists

        safety_move, safe_moves = self._safety_net_choices(state, valid_moves)
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
    
    

    def _safety_net_choices(self, state, valid_moves):
        """Return forced move if present and list of moves that keep the opponent from an immediate win."""
        board, current_player = state
        board_arr = np.array(board, copy=True)
        forced_move, safe_moves = GameEnv.safety_net_choices(board_arr, current_player, valid_moves)
        return forced_move, safe_moves

    # def _finalize_episode(self, outcome_reward: float):
    #     if not self._episode_transitions:
    #         return

    #     return_to_go = float(outcome_reward)
    #     for transition in reversed(self._episode_transitions):
    #         state = transition["state"]
    #         action = transition["action"]
    #         state_h = self._make_hashable(state)
    #         state_q = self.q_values.setdefault(state_h, defaultdict(float))
    #         current_q = state_q.get(action, 0.0)
    #         state_q[action] = current_q + self.lr * (return_to_go - current_q)
    #         return_to_go *= self.gamma

    #     self._episode_transitions.clear()
