from typing import Sequence
import random

import numpy as np

from agents.base_agent import BaseAgent
from game.environment import GameEnv


class PureRandomAgent(BaseAgent):
    def __init__(self, role):
        super().__init__(role)

    @property
    def name(self):
        return "PureRandomAgent"

    def compute_reward(self, last_state, action: tuple[int, int] | None, new_state, winner) -> float:
        # Pure random agent does not learn, so reward is always 0
        return 0.0

    def choose_action(
        self,
        state,
        valid_moves: Sequence[tuple[int, int]],
        learn: bool = True,
        *,
        use_safety_net: bool = False,
    ) -> tuple[int, int]:
        if not valid_moves:
            raise ValueError("valid_moves must contain at least one move")

        if use_safety_net:
            board, current_player = state
            board_arr = np.array(board, copy=True)
            winning_move, safe_moves = GameEnv.safety_net_choices(board_arr, current_player, list(valid_moves))
            if winning_move is not None:
                return winning_move
            if safe_moves:
                return random.choice(safe_moves)

        # fall back to full randomness
        return random.choice(valid_moves)