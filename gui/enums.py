from enum import Enum

class GameMode(Enum):
    BOT_V_BOT = "bot vs bot"
    PLAYER_V_PLAYER = "player vs player"
    PLAYER_V_BOT = "player vs bot"
    VIEW_MATCH = "view match"
