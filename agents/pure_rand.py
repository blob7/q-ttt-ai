from agents.base_agent import BaseAgent
import random

class PureRandomAgent(BaseAgent):
    def __init__(self, role):
        super().__init__(role)

    @property
    def name(self):
        return "PureRandomAgent"

    def compute_reward(self, last_state, action: tuple[int, int], new_state, winner) -> float:
        # Pure random agent does not learn, so reward is always 0
        return 0.0
    
    def choose_action(self, state, valid_moves):
        choice = random.choice(valid_moves)
        return choice