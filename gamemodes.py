from enum import Enum

class GameMode(Enum):
    BOT_V_BOT = "bot vs bot"
    PLAYER_V_PLAYER = "player vs player"
    PLAYER_V_BOT = "player vs bot"
    VIEW_MATCH = "view match"
    TRAIN_AI = "train ai"
    COMPETE_BOT_VS_BOT = "compete bot vs bot"