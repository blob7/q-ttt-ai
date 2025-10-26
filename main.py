from gui.game import TicTacToeGUI, GameMode
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
            gui = TicTacToeGUI(mode=GameMode.PLAYER_V_BOT, bot1=random_bot)
        case "2":
            gui = TicTacToeGUI(mode=GameMode.PLAYER_V_PLAYER)
        case "3":
            gui = TicTacToeGUI(mode=GameMode.BOT_V_BOT, bot1=random_bot, bot2=random_bot)
        case "4":
            print("Training mode not implemented yet")
            return
        case "5":
            print("View Match not implemented yet")
            return
        case _:
            print("Invalid choice")
            return
    gui.run()



def random_bot(env):
    moves = env.get_valid_moves()
    return random.choice(moves)


if __name__ == "__main__":
    main()