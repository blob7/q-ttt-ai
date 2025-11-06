from abc import ABC, abstractmethod
from collections import defaultdict
import random
import pickle
from pathlib import Path
import numpy as np
from game.environment import GameEnv

class BaseAgent(ABC):   
    def __init__(self, role = None, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.2):
        self.role = role
        self.q_values = defaultdict(lambda: defaultdict(float))  # Q-table
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Must be implemented by each agent.
        Returns the name/identifier of the agent.
        """
        pass

    # --- Core Q-learning update ---
    def update_q_value(self, last_state, action: tuple[int, int], new_state, reward: float, done: bool):
        """
        Update Q-value for the last_state/action pair based on reward and new_state.
        """
        # Convert numpy arrays or other mutable types to tuples for dict keys
        last_state_h = self._make_hashable(last_state)
        new_state_h = self._make_hashable(new_state)
        # Initialize Q-values for all empty positions in the new state
        if action not in self.q_values[last_state_h]:
            self.q_values[last_state_h][action] = 0.0

        # Compute target Q-value
        max_future_q = 0.0 if done else max(self.q_values[new_state_h].values(), default=0.0)
        target = reward if done else reward + self.gamma * max_future_q

        # Update Q-value for last state/action
        self.q_values[last_state_h][action] += self.lr * (target - self.q_values[last_state_h][action])

    # --- Action selection (epsilon-greedy) ---
    def choose_action(self, state, valid_moves, learn: bool = True):
        """Select action using epsilon-greedy strategy with lazy Q-value initialization."""
        state_h = self._make_hashable(state)
        state_q = self.q_values.setdefault(state_h, defaultdict(float))  # ensure dict exists

        safety_move, safe_moves = self._safety_net_choices(state, valid_moves)
        if safety_move is not None:
            return safety_move

        eligible_moves = safe_moves if safe_moves else valid_moves

        # Epsilon-greedy policy
        if learn and random.random() < self.epsilon:
            # Explore: random valid move
            return random.choice(eligible_moves)

        # Exploit: pick move with max Q-value (default to 0.0 if unseen)
        best_move = None
        best_value = float('-inf')
        for move in eligible_moves:
            q = state_q.get(move, 0.0)  # lazy read
            if q > best_value:
                best_move = move
                best_value = q
        # Fallback (shouldn't happen unless valid_moves empty)
        return best_move if best_move is not None else random.choice(valid_moves)
            
    # --- Reward computation ---
    @abstractmethod
    def compute_reward(self, last_state, action, new_state, winner) -> float:
        """
        Must be implemented by each agent.
        Determines the reward for the last action based on agent's own logic.
        """
        pass

    # --- Combined learning step ---
    def learn(self, last_state, action: tuple[int, int] | None, new_state, done: bool, winner):
        reward = self.compute_reward(last_state, action, new_state, winner)
        if action is None:
            return
        self.update_q_value(last_state, action, new_state, reward, done)

    def _make_hashable(self, state):
        board, player = state

        # Ensure board is numpy array
        board = np.array(board, dtype=int)

        # Use canonical version to reduce symmetry
        canonical_board = self._canonical_board(board)

        # Flatten and encode to compact string
        flat = [cell for row in canonical_board for cell in row]
        encoded = ''.join(str(cell if cell >= 0 else 2) for cell in flat)

        return (encoded, int(player))


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
    
    
    def _canonical_board(self, board):
        """
        Returns the lexicographically smallest board among all rotations and mirror flips.
        This ensures symmetric states share the same hash, reducing Q-table size.
        """
        rotations = [np.rot90(board, k) for k in range(4)]
        mirrors = [np.fliplr(r) for r in rotations]
        all_forms = rotations + mirrors
        # Convert each to tuple-of-tuples for comparison
        return min(tuple(map(tuple, b)) for b in all_forms)


    def _safety_net_choices(self, state, valid_moves):
        """Return forced move if present and list of moves that keep the opponent from an immediate win."""
        board, current_player = state
        board_arr = np.array(board, copy=True)
        forced_move, safe_moves = GameEnv.safety_net_choices(board_arr, current_player, valid_moves)
        return forced_move, safe_moves