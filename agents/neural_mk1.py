# depricated until memory becomes a problem

import random
from math import prod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from agents.base_agent import BaseAgent
from game.environment import GameEnv
from game.board import TicTacToe9x9


class QNetwork(nn.Module):
    def __init__(self, board_cells: int, extra_features: int = 2, hidden_sizes: tuple[int, ...] = (256, 128)) -> None:
        raise NotImplementedError("NeuralQAgent is not yet implemented.")
        super().__init__()
        input_dim = board_cells + extra_features
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, board_cells))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NeuralQAgent(BaseAgent):
    def __init__(
        self,
        role: int,
        *,
        board_shape: tuple[int, int] = (9, 9),
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        network_lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        super().__init__(role, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon)
        self.board_shape = board_shape
        self.board_cells = prod(board_shape)
        target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(target_device)
        self.network = QNetwork(self.board_cells).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=network_lr)

    @property
    def name(self):
        return "NeuralQAgent"


    def choose_action(self, state, valid_moves, learn: bool = True):
        board, current_player = state
        board_arr = np.array(board, copy=True)

        forced_move, safe_moves = GameEnv.safety_net_choices(board_arr, current_player, valid_moves)
        if forced_move is not None:
            return forced_move

        candidate_moves = safe_moves if safe_moves else valid_moves
        if learn and random.random() < self.epsilon:
            return random.choice(candidate_moves)

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.network(state_tensor).view(-1).cpu().numpy()

        best_move = None
        best_value = float("-inf")
        for move in candidate_moves:
            idx = self._move_to_index(move)
            q = q_values[idx]
            if q > best_value:
                best_value = q
                best_move = move

        return best_move if best_move is not None else random.choice(candidate_moves)

    def compute_reward(self, last_state, action, new_state, winner) -> float:
        if winner == self.role:
            return 1.0
        if winner is not None and winner != 0:
            return -1.0
        return 0.0

    def learn(self, last_state, action, new_state, done: bool, winner) -> None:
        if action is None:
            return

        reward = self.compute_reward(last_state, action, new_state, winner)

        state_tensor = self._state_to_tensor(last_state)
        self.optimizer.zero_grad()
        q_values = self.network(state_tensor)
        action_index = self._move_to_index(action)
        predicted_q = q_values[0, action_index]

        target_value = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if not done:
            next_state_tensor = self._state_to_tensor(new_state)
            with torch.no_grad():
                next_q_values = self.network(next_state_tensor).view(-1)
            future_moves = self._valid_moves_after(action, new_state)
            if future_moves:
                future_indices = [self._move_to_index(mv) for mv in future_moves]
                max_future_q = next_q_values[future_indices].max()
                target_value = target_value + self.gamma * max_future_q

        loss = F.mse_loss(predicted_q, target_value.detach())
        loss.backward()
        self.optimizer.step()

    def learn_result(self, winner, final_state):
        # Neural agent trains incrementally per move; no episodic update needed.
        return

    def _state_to_tensor(self, state) -> torch.Tensor:
        board, current_player = state
        board_arr = np.array(board, dtype=np.float32).reshape(-1)
        extras = np.array([current_player, self.role], dtype=np.float32)
        features = np.concatenate([board_arr, extras])[None, :]
        tensor = torch.from_numpy(features).to(self.device)
        return tensor

    def _move_to_index(self, move: tuple[int, int]) -> int:
        rows, cols = self.board_shape
        r, c = move
        return r * cols + c

    def _valid_moves_after(self, action: tuple[int, int], state) -> list[tuple[int, int]]:
        board, current_player = state
        temp_game = TicTacToe9x9()
        temp_game.board = np.array(board, dtype=int)
        temp_game.current_player = current_player
        temp_game.last_move = action
        return temp_game.get_valid_moves()


