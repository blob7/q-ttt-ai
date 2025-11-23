# game/utils.py


def print_board(board):
    """Pretty-print the board in console."""
    symbols = {1: "X", -1: "O", 0: "."}
    print("\n".join(" ".join(symbols[v] for v in row) for row in board))
    print()


def encode_move(move, board_size: int) -> int:
    """Pack a board move into a single integer (supports sizes < 256)."""
    if move is None:
        return -1
    row, col = move
    if row >= board_size or col >= board_size:
        raise ValueError(f"Move {(row, col)} outside board size {board_size}")
    return (row << 8) | col


def decode_move(encoded: int, board_size: int):
    """Unpack an integer move back into a (row, col) tuple."""
    if encoded < 0:
        return None
    row = (encoded >> 8) & 0xFF
    col = encoded & 0xFF
    if row >= board_size or col >= board_size:
        raise ValueError(f"Decoded move {(row, col)} exceeds board size {board_size}")
    return (row, col)
