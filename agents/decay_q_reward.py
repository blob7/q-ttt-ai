from agents.base_agent import BaseAgent


class SimpleDecayTicTacToeAgent(BaseAgent):
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
        return "SimpleDecayTicTacToeAgent"


    
    def compute_reward(self, state, action, winner, mover, turn: int) -> float:
        # 0.2 for early start move if long game 1 for winnig move
        if winner == mover:
            return 1.0
        if winner in (None, 0):
            return 0.0
        return -1.0
