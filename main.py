from typing import Optional
from agents.base_agent import BaseAgent
from gui.game import TicTacToeGUI, GameMode

from game.board import PlayerPiece
from game.environment import GameEnv

from gamemodes import GameMode

from agents import *

from training.train import train_agents



import cProfile
import pstats

import random


# def main():
#     print("1. Play vs Bot")
#     print("2. Play vs Human")
#     print("3. Watch Bot vs Bot")
#     print("4. Train AI")
#     print("5. View Match")
#     choice = input("> ")

#     match choice:
#         case "1":
#             print("playing against bot")
#             if False:
#                 random_bot = PureRandomAgent(role=PlayerPiece.O)
#                 bot = lambda env: random_bot.choose_action(env, False, use_safety_net=True)
#                 gui = TicTacToeGUI(mode=GameMode.PLAYER_V_BOT, bot2=bot)
#             else:
#                 simple_agent = SimpleTicTacToeAgent.load("data/saved_agents/simple_agent_100k.pkl", role=PlayerPiece.O.value)
#                 bot = lambda env: simple_agent.choose_action(env, False)
#                 # print(simple_agent.epsilon, simple_agent.learning_rate, simple_agent.discount_factor)
#                 # print(simple_agent.q_values)
#                 gui = TicTacToeGUI(mode=GameMode.PLAYER_V_BOT, bot2=bot)
#         case "2":
#             gui = TicTacToeGUI(mode=GameMode.PLAYER_V_PLAYER)
#         case "3":
#             random_bot = PureRandomAgent(role=PlayerPiece.X)
#             gui = TicTacToeGUI(mode=GameMode.BOT_V_BOT, bot1=random_bot.choose_action, bot2=random_bot.choose_action)
#         case "4":
#             # profiler = cProfile.Profile()
#             # profiler.enable()
#             simple_agent_x = SimpleTicTacToeAgent(role=PlayerPiece.X.value)
#             simple_agent_o = SimpleTicTacToeAgent(role=PlayerPiece.O.value)
#             env = GameEnv()
#             train_agents(
#                 env=env, 
#                 agent_x=simple_agent_x, 
#                 agent_o=simple_agent_o, 
#                 episodes=100_000, 
#                 memory_stop_threshold_mb=15_000, 
#                 agent_o_save_path="data/saved_agents/simple_agent_100k.pkl", 
#                 show_progress=True,
#                 coin_flip_start=True,
#                 parallel=False,
#             )
#             # profiler.disable()
#             # stats = pstats.Stats(profiler).sort_stats('cumtime')  # sort by cumulative time
#             # stats.print_stats(20)  # top 20 functions
#             return
#         case "5":
#             gui = TicTacToeGUI(mode=GameMode.VIEW_MATCH)
#         case _:
#             print("Invalid choice")
#             return
#     gui.run()

import menus.game_setup as menu

def main():
    game_mode = menu.game_setup_menu()
    player1 = None
    bot1 = None
    controller1 = None
    player2 = None
    bot2 = None
    controller2 = None
    do_learning = False
    coinflip_start = False
    starting_player = None

    if game_mode in (GameMode.PLAYER_V_BOT, GameMode.BOT_V_BOT, GameMode.TRAIN_AI, GameMode.COMPETE_BOT_VS_BOT):
        player1 = menu.select_agent_menu()
        bot1 = load_agent(player1, role=PlayerPiece.X.value, path=menu.ask_file_path() if menu.select_agent_load_menu() == menu.AgentLoadChoice.FROM_FILE else None)
        
        if game_mode in (GameMode.BOT_V_BOT, GameMode.TRAIN_AI, GameMode.COMPETE_BOT_VS_BOT):
            player2 = menu.select_agent_menu()
            bot2 = load_agent(player2, role=PlayerPiece.O.value, path=menu.ask_file_path() if menu.select_agent_load_menu() == menu.AgentLoadChoice.FROM_FILE else None)
            
            controller1 = lambda env: bot1.choose_action(env, learn=game_mode == GameMode.TRAIN_AI)
            controller2 = lambda env: bot2.choose_action(env, learn=game_mode == GameMode.TRAIN_AI)
        
        
    
        starting_player = menu.select_starting_player_menu(
            bot1.name, 
            bot2.name if bot2 else "Human"
        )
        coinflip_start = starting_player == "Random (coin flip)"

    if game_mode in (GameMode.PLAYER_V_BOT, GameMode.BOT_V_BOT):
        if coinflip_start:
            starting_player = random.choice([controller1, controller2])
            second_player = controller2 if starting_player == controller1 else controller1
        elif bot1:
            starting_player = controller1 if starting_player == bot1.name else controller2
            second_player = controller2 if starting_player == controller1 else controller1
        gui = TicTacToeGUI(mode=game_mode, bot1=controller1, bot2=controller2)
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
                parallel=False,
            )
        elif game_mode == GameMode.COMPETE_BOT_VS_BOT:
            competition_params = menu.select_competition_parameters_menu()

            print('UNIMPLEMENTED: Bot vs Bot competition mode is not yet implemented.')
            # competition_results = compete_bots(
            #     env=env,
            #     bot1=bot1,
            #     bot2=bot2,
            #     episodes=competition_params["episodes"],
            #     coin_flip_start=coinflip_start,
            # )
            # print(f"Competition results: {competition_results}")



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
    main()