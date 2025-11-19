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

def select_agent_menu() -> AgentChoice:
    choice = q.select(
        "Select an agent:",
        choices=[agent.value for agent in AgentChoice]
    ).ask()
    return AgentChoice(choice)

class AgentLoadChoice(Enum):
    FROM_FILE = "Load from file"
    FRESH = "Start fresh"

def select_agent_load_menu() -> AgentLoadChoice:
    choice = q.select(
        "Do you want to load an existing agent or start fresh?",
        choices=[option.value for option in AgentLoadChoice]
    ).ask()
    return AgentLoadChoice(choice)

def ask_file_path() -> str:
    path = q.text("Enter the file path to load the agent from:", default="data/saved_agents/simple_agent_100k.pkl").ask()
    return path

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

def select_training_parameters_menu() -> dict:
    episodes = q.text("Enter the number of training episodes:", default="10_0000").ask()
    memory_threshold = q.text("Enter memory stop threshold in MB:", default="15_000").ask()
    save_agents = q.confirm("Do you want to save the trained agents?").ask()
    if save_agents:
        agent_x_save_path = None
        agent_o_save_path = None
    else:
        agent_x_save_path = q.text("Enter the file path to save the trained agent X:", default="data/saved_agents/agent_x_trained.pkl").ask()
        agent_o_save_path = q.text("Enter the file path to save the trained agent O:", default="data/saved_agents/agent_o_trained.pkl").ask()
    show_progress = q.confirm("Do you want to show training progress?").ask()
    
    
    return {
        "episodes": int(episodes),
        "memory_stop_threshold_mb": int(memory_threshold),
        "agent_x_save_path": agent_x_save_path,
        "agent_o_save_path": agent_o_save_path,
        "show_progress": show_progress,
    }

def select_competition_parameters_menu() -> dict:
    num_games = q.text("Enter the number of games for the competition:", default="10_000").ask()
    return {
        "num_games": int(num_games)
    }