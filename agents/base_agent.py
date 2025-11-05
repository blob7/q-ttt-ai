from abc import ABC, abstractmethod
from collections import defaultdict
import random
import pickle
from pathlib import Path

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
    def update_q_value(self, last_state, action: tuple[int, int] | None, new_state, reward: float, done: bool):
        """
        Update Q-value for the last_state/action pair based on reward and new_state.
        """
        # Convert numpy arrays or other mutable types to tuples for dict keys
        last_state_h = self._make_hashable(last_state)
        new_state_h = self._make_hashable(new_state)

        board_tuple, _ = new_state_h 
        # Initialize Q-values for all empty positions in the new state
        for i, row in enumerate(board_tuple):
            for j, cell in enumerate(row):
                if cell == 0 and (i, j) not in self.q_values[new_state_h]:
                    self.q_values[new_state_h][(i, j)] = 0.0

        # Compute target Q-value
        max_future_q = 0.0 if done else max(self.q_values[new_state_h].values(), default=0.0)
        target = reward if done else reward + self.gamma * max_future_q

        # Update Q-value for last state/action
        self.q_values[last_state_h][action] += self.lr * (target - self.q_values[last_state_h][action])

    # --- Action selection (epsilon-greedy) ---
    def choose_action(self, state, valid_moves, learn: bool = True):
        """
        Select action using epsilon-greedy.
        Converts state to hashable tuple-of-tuples.
        """
        state_h = self._make_hashable(state)

        if learn:
            # Initialize Q-values for unseen moves
            for move in valid_moves:
                if move not in self.q_values[state_h]:
                    self.q_values[state_h][move] = 0.0
            # Epsilon-greedy choice
            if random.random() < self.epsilon:
                return random.choice(valid_moves)
            return max(valid_moves, key=lambda a: self.q_values[state_h][a])
        else:
            for move in valid_moves:
                if move not in self.q_values[state_h]:
                    self.q_values[state_h][move] = 0.0
            return max(valid_moves, key=lambda a: self.q_values[state_h][a])
            
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
        reward = self.compute_reward(last_state, action, new_state, done)
        self.update_q_value(last_state, action, new_state, reward, done)

    def _make_hashable(self, state):
        board, player = state  # state = (board, current_player)
        
        # Ensure board is tuple-of-tuples of ints
        board_tuple = tuple(
            tuple(int(cell) for cell in row) 
            for row in (board.tolist() if hasattr(board, "tolist") else board)
        )
        return (board_tuple, int(player))




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