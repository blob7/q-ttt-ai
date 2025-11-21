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

    
    def compute_reward(
        self,
        state,
        action,
        winner: int | None,
        mover: int,
        steps_from_end: int
    ) -> float:
        """Symmetric decayed rewards for both wins and losses, clamped at floors."""
        min_win: float = 0.2  # floor for wins
        max_loss: float = -0.2  # floor (ceiling) for losses
    
        # --- Draw case ---
        if winner in (None, 0):
            return 0.0
    
        # --- Shared decay for win/loss ---
        decay: float = self.discount_factor ** steps_from_end   # 0 steps → 1.0
    
    
        if winner == mover:
            return round(max(min_win, decay), 3)
        else:
            return round(min(max_loss, -decay), 3)


