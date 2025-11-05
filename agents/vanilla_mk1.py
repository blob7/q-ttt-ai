from agents.base_agent import BaseAgent

class SimpleTicTacToeAgent(BaseAgent):
    def __init__(self, role):
        super().__init__(role)

    @property
    def name(self):
        return "SimpleTicTacToeAgent"

    def compute_reward(self, last_state, action: tuple[int, int], new_state, winner) -> float:
        # Example: +1 if win, -1 if loss, 0 otherwise
        if winner == self.role:
            return 1
        if winner is not None:
            return -1
        return 0
