from typing import Any, Dict, List, Optional, Tuple
from game.board import PlayerPiece
from game.environment import GameEnv
from agents.base_agent import BaseAgent
import random

def run_episode(env: GameEnv, agent_x: BaseAgent, agent_o: BaseAgent, coin_flip_start: bool = False, record_stats: bool = False) -> Tuple[List[Any], Optional[int], Optional[Dict[str, Any]]]:
    """Run a single episode and return the winner."""
    stats: Optional[Dict[str, Any]] = None

    state = env.reset()
    done = False
    winner: Optional[int] = None

    first_agent, second_agent = determine_starting_player(agent_x, agent_o, coin_flip_start)
    first_agent.role = PlayerPiece.X.value
    second_agent.role = PlayerPiece.O.value

    while not done:
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            done = True
            winner = None
            break

        action = first_agent.choose_action(env)
        new_state, done, winner = env.step(action)
        if winner is not None:
            if record_stats:
                stats = {
                    'winning_player': first_agent.name,
                    'winning_piece': first_agent.role,
                    'turns': env.game.turn_count,
                    'last_move': action
                }
            return env.state_history, winner, stats

        first_agent, second_agent = second_agent, first_agent
    
    if record_stats and winner is None:
        stats = {
            'winning_player': None,
            'winning_piece': None,
            'turns': env.game.turn_count,
            'last_move': env.game.last_move
        }
    return env.state_history, None, stats


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