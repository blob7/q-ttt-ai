from agents.base_agent import BaseAgent


class SimpleTicTacToeAgent(BaseAgent):
    def __init__(
        self,
        role: int,
        *,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
    ) -> None:
        super().__init__(
            role=role,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )

    @property
    def name(self) -> str:
        return "SimpleTicTacToeAgent"
    
    
    def compute_reward(self, state, action, winner, mover, steps_from_end: int) -> float:
        if winner == mover:
            return 1.0
        if winner in (None, 0):
            return 0.0
        return -1.0
