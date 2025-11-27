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

    def choose_action(
        self, env: GameEnv, learn: bool = True, *, use_safety_net: bool = True
    ):
        if use_safety_net:
            safety_move, safe_moves = env.safety_net_choices()
            if safety_move is not None:
                return safety_move
            elif safe_moves:
                return random.choice(safe_moves)
        
        valid_moves = env.get_valid_moves()
        return random.choice(valid_moves)
    
    def compute_reward(self, state, action, winner: int | None, mover, steps_from_end: int) -> float:
        return 0.0  # Pure random agent does not learn, so reward is always 0