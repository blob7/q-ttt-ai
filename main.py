from gui.game import TicTacToeGUI, GameMode

from game.board import PlayerPiece
from game.environment import GameEnv


from agents import *

from training.train import train_agents



import cProfile
import pstats

def main():
    print("1. Play vs Bot")
    print("2. Play vs Human")
    print("3. Watch Bot vs Bot")
    print("4. Train AI")
    print("5. View Match")
    choice = input("> ")

    match choice:
        case "1":
            print("playing against bot")
            if False:
                random_bot = PureRandomAgent(role=PlayerPiece.O)
                bot = lambda state, moves: random_bot.choose_action(state, moves, False, use_safety_net=True)
                gui = TicTacToeGUI(mode=GameMode.PLAYER_V_BOT, bot2=bot)
            else:
                simple_agent = SimpleTicTacToeAgent.load("data/saved_agents/simple_agent_X.pkl")
                print(f'all q_values: {simple_agent.q_values}')
                bot = lambda state, state_hash, moves: simple_agent.choose_action(state, state_hash, moves, False)
                gui = TicTacToeGUI(mode=GameMode.PLAYER_V_BOT, bot1=bot)
        case "2":
            gui = TicTacToeGUI(mode=GameMode.PLAYER_V_PLAYER)
        case "3":
            random_bot = PureRandomAgent(role=PlayerPiece.X)
            gui = TicTacToeGUI(mode=GameMode.BOT_V_BOT, bot1=random_bot.choose_action, bot2=random_bot.choose_action)
        case "4":
            profiler = cProfile.Profile()
            profiler.enable()
            simple_agent_x = SimpleTicTacToeAgent(role=PlayerPiece.X.value)
            simple_agent_o = SimpleTicTacToeAgent(role=PlayerPiece.O.value)
            env = GameEnv()
            train_agents(
                env=env, 
                agent_x=simple_agent_x, 
                agent_o=simple_agent_o, 
                episodes=1000, 
                memory_stop_threshold_mb=15_000, 
                agent_o_save_path="data/saved_agents/simple_agent_o.pkl", 
                show_progress=True,
                coin_flip_start=True
            )
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumtime')  # sort by cumulative time
            stats.print_stats(20)  # top 20 functions
            return
        case "5":
            gui = TicTacToeGUI(mode=GameMode.VIEW_MATCH)
        case _:
            print("Invalid choice")
            return
    gui.run()




if __name__ == "__main__":
    main()