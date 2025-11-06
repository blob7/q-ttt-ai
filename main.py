from gui.game import TicTacToeGUI, GameMode

from game.board import PlayerPiece

from agents import *

from agents.train import train_agents

from game.environment import GameEnv

import random

def main():
    print("1. Play vs Bot")
    print("2. Play vs Human")
    print("3. Watch Bot vs Bot")
    print("4. Train AI")
    print("5. View Match")
    choice = input("> ")

    match choice:
        case "1":
            if False:
                random_bot = PureRandomAgent(role=PlayerPiece.O)
                bot = lambda state, moves: random_bot.choose_action(state, moves, False, use_safety_net=True)
                gui = TicTacToeGUI(mode=GameMode.PLAYER_V_BOT, bot2=bot)
            else:
                simple_agent = SimpleTicTacToeAgent.load("data/saved_agents/simple_agent_o.pkl")
                bot = lambda state, moves: simple_agent.choose_action(state, moves, False)
                gui = TicTacToeGUI(mode=GameMode.PLAYER_V_BOT, bot2=bot)
        case "2":
            gui = TicTacToeGUI(mode=GameMode.PLAYER_V_PLAYER)
        case "3":
            random_bot = PureRandomAgent(role=PlayerPiece.X)
            gui = TicTacToeGUI(mode=GameMode.BOT_V_BOT, bot1=random_bot.choose_action, bot2=random_bot.choose_action)
        case "4":
            simple_agent_x = SimpleTicTacToeAgent(role=PlayerPiece.X.value)
            simple_agent_o = SimpleTicTacToeAgent(role=PlayerPiece.O.value)
            env = GameEnv()
            train_agents(
                env=env, 
                agent_x=simple_agent_x, 
                agent_o=simple_agent_o, 
                episodes=1_000, 
                memory_stop_threshold_mb=18_000, 
                agent_o_save_path="data/saved_agents/simple_agent_o.pkl", 
                agent_x_save_path="data/saved_agents/simple_agent_x.pkl",
                show_progress=True
            )
            return
        case "5":
            gui = TicTacToeGUI(mode=GameMode.VIEW_MATCH)
        case _:
            print("Invalid choice")
            return
    gui.run()




if __name__ == "__main__":
    main()