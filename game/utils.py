# game/utils.py
def print_board(board):
    """Pretty-print the board in console."""
    symbols = {1: "X", -1: "O", 0: "."}
    print("\n".join(" ".join(symbols[v] for v in row) for row in board))
    print()
