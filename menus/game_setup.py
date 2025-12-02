from typing import Optional, Tuple
import questionary as q

from enum import Enum

from gamemodes import GameMode


def game_setup_menu() -> GameMode:
    choice = q.select(
        "Select a game mode:",
        choices=[mode.value for mode in GameMode]
    ).ask()
    return GameMode(choice)


class AgentChoice(Enum):
    SIMPLE_Q_AGENT = "Simple Q-Learning Agent"
    DECAY_Q_AGENT = "Decay Q-Learning Agent"
    PURE_RANDOM_AGENT = "Pure Random Agent"


def ask_self_play_training() -> bool:
    return q.confirm("Train the first agent against itself?").ask()


def select_agent_menu() -> Tuple[AgentChoice, Optional[str]]:
    choice = q.select(
        "Select an agent:",
        choices=[agent.value for agent in AgentChoice]
    ).ask()
    path = q.text("Enter the name of the file to load the agent from (leave blank for fresh agent):", default="").ask()
    path = "data/saved_agents/" + path + ".pkl" if path else None
    return AgentChoice(choice), path

def select_starting_player_menu(player1: str, player2: str) -> str:
    choice = q.select(
        "Who should start first? (X)",
        choices=[
            player1,
            player2,
            "Random (coin flip)"
        ]
    ).ask()
    return choice

def select_training_parameters_menu(self_play: bool = False) -> dict:
    episodes = q.text("Enter the number of training episodes:", default="100_000").ask()
    memory_threshold = q.text("Enter memory stop threshold in MB:", default="12_000").ask()
    
    agent_x_save_path = q.text("Enter the file name to save the first trained agent (leave blank for none):", default="").ask()
    agent_x_save_path = "data/saved_agents/" + agent_x_save_path + ".pkl" if agent_x_save_path else None
    if self_play:
        agent_o_save_path = None
    else:
        agent_o_save_path = q.text("Enter the file name to save the second trained agent (leave blank for none):", default="").ask()
        agent_o_save_path = "data/saved_agents/" + agent_o_save_path + ".pkl" if agent_o_save_path else None
    
    show_progress = q.confirm("Do you want to show training progress?").ask()
    parallel = q.confirm("Do you want to train in parallel?").ask()
    
    def _parse_int(value: str) -> int:
        return int(value.replace("_", "").strip())

    return {
        "episodes": _parse_int(episodes),
        "memory_stop_threshold_mb": _parse_int(memory_threshold),
        "agent_x_save_path": agent_x_save_path,
        "agent_o_save_path": agent_o_save_path,
        "show_progress": show_progress,
        "parallel": parallel,
        "self_play": self_play,
    }

def select_competition_parameters_menu() -> dict:
    episodes = q.text("Enter the number of games for the competition:", default="1_000").ask()
    show_progress = q.confirm("Do you want to show competition progress?").ask()
    visualize = q.confirm("Do you want to visualize the competition dashboard?").ask()
    bot1_name = q.text("Enter the name for Bot 1 (leave blank for default):", default="").ask()
    bot2_name = q.text("Enter the name for Bot 2 (leave blank for default):", default="").ask()
    
    
    return {
        "episodes": int(episodes),
        "show_progress": show_progress,
        "visualize": visualize,
        "bot1_name": bot1_name if bot1_name else None,
        "bot2_name": bot2_name if bot2_name else None,
    }