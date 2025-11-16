from typing import Any, List, Optional, Tuple
from game.environment import GameEnv
from agents.base_agent import BaseAgent
import random

def run_episode(env: GameEnv, agent_x: BaseAgent, agent_o: BaseAgent, coin_flip_start: bool = False) -> Tuple[List[Any], Optional[int]]:
    """Run a single episode and return the winner."""

    state = env.reset()
    done = False
    winner: Optional[int] = None

    first_agent, second_agent = determine_starting_player(agent_x, agent_o, coin_flip_start)

    while not done:
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            done = True
            winner = None
            break

        action = first_agent.choose_action(env)
        new_state, done, winner = env.step(action)
        if winner is not None:
            return env.state_history, winner

        state = new_state
        first_agent, second_agent = second_agent, first_agent
    
    return env.state_history, None

def determine_starting_player(agent_x: BaseAgent, agent_o: BaseAgent, coin_flip_start: bool) -> tuple[BaseAgent, BaseAgent]:
    """Determine the starting player based on coin flip and agent roles."""
    first_agent = agent_x
    second_agent = agent_o

    if coin_flip_start:
        if random.random() < 0.5:
            first_agent, second_agent = second_agent, first_agent
        desired = getattr(first_agent, "role", None)
        if desired in (1, -1):
            return first_agent, second_agent

    return first_agent, second_agent