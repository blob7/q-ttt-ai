from typing import Optional
from agents.base_agent import BaseAgent
from gui.game import TicTacToeGUI, GameMode

from game.board import PlayerPiece
from game.environment import GameEnv

from gamemodes import GameMode

from agents import *

from agent_play.train import train_agents
from agent_play.competition import compete_bots


import cProfile
import pstats

import random


import menus.game_setup as menu

def main():
    game_mode = menu.game_setup_menu()
    player1 = None
    bot1 = None
    controller1 = None
    player2 = None
    bot2 = None
    controller2 = None
    coinflip_start = False
    starting_player = None

    if game_mode in (GameMode.PLAYER_V_BOT, GameMode.BOT_V_BOT, GameMode.TRAIN_AI, GameMode.COMPETE_BOT_VS_BOT):
        player1, path1 = menu.select_agent_menu()
        bot1 = load_agent(player1, role=PlayerPiece.X.value, path=path1)
        controller1 = lambda env: bot1.choose_action(env, learn=game_mode == GameMode.TRAIN_AI)
        
        if game_mode in (GameMode.BOT_V_BOT, GameMode.TRAIN_AI, GameMode.COMPETE_BOT_VS_BOT):
            player2, path2 = menu.select_agent_menu()
            bot2 = load_agent(player2, role=PlayerPiece.O.value, path=path2)
            controller2 = lambda env: bot2.choose_action(env, learn=game_mode == GameMode.TRAIN_AI)

    
        starting_player = menu.select_starting_player_menu(
            bot1.name, 
            bot2.name if bot2 else "Human"
        )
        coinflip_start = starting_player == "Random (coin flip)"

    if game_mode in (GameMode.PLAYER_V_BOT, GameMode.BOT_V_BOT):
        if coinflip_start:
            first_player = random.choice([controller1, controller2])
            second_player = controller2 if first_player == controller1 else controller1
        else:
            if not bot1:
                raise ValueError("Bot1 must be defined for PLAYER_V_BOT or BOT_V_BOT modes.")
            first_player = controller1 if starting_player == bot1.name else controller2
            second_player = controller2 if starting_player == bot1.name else controller1
        
        gui = TicTacToeGUI(mode=game_mode, bot1=first_player, bot2=second_player)
        gui.run()
    elif game_mode == GameMode.PLAYER_V_PLAYER:
        gui = TicTacToeGUI(mode=game_mode)
        gui.run()
    elif game_mode == GameMode.VIEW_MATCH:
        TicTacToeGUI(mode=game_mode).run()
    elif game_mode in (GameMode.TRAIN_AI, GameMode.COMPETE_BOT_VS_BOT) and bot1 is not None and bot2 is not None:
        # Non-GUI setup for training or competition
        env = GameEnv()
        if game_mode == GameMode.TRAIN_AI:
            training_params = menu.select_training_parameters_menu()
            train_agents(
                env=env, 
                agent_x=bot1, 
                agent_o=bot2, 
                episodes=training_params["episodes"], 
                memory_stop_threshold_mb=training_params["memory_stop_threshold_mb"], 
                agent_x_save_path=training_params["agent_x_save_path"],
                agent_o_save_path=training_params["agent_o_save_path"],
                show_progress=training_params["show_progress"],
                coin_flip_start=coinflip_start,
                parallel=training_params['parallel'],
            )
        elif game_mode == GameMode.COMPETE_BOT_VS_BOT:
            competition_params = menu.select_competition_parameters_menu()
            compete_bots(
                env=env,
                bot1=bot1,
                bot2=bot2,
                episodes=competition_params["episodes"],
                coin_flip_start=coinflip_start,
                show_progress=competition_params["show_progress"],
                visualize=competition_params["visualize"],
                bot1_name=competition_params.get("bot1_name", bot1.name),
                bot2_name=competition_params.get("bot2_name", bot2.name),
            )


def load_agent(agent_choice: menu.AgentChoice, role: int, path: Optional[str]) -> BaseAgent:
    if agent_choice == menu.AgentChoice.SIMPLE_Q_AGENT:
        return SimpleTicTacToeAgent.load(path, role=role) if path else SimpleTicTacToeAgent(role=role)
    elif agent_choice == menu.AgentChoice.DECAY_Q_AGENT:
        return SimpleDecayTicTacToeAgent.load(path, role=role) if path else SimpleDecayTicTacToeAgent(role=role)
    elif agent_choice == menu.AgentChoice.PURE_RANDOM_AGENT:
        return PureRandomAgent(role=role)
    else:
        raise ValueError(f"Invalid agent choice: {agent_choice}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)  # Print top 20 functions by cumulative time