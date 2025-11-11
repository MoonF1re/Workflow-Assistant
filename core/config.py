from pathlib import Path

# базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# пути
DATA_DIR = BASE_DIR / "data"
COMMANDS_DIR = BASE_DIR / "commands"
LOGS_DIR = BASE_DIR / "logs"
SOUNDS_DIR = DATA_DIR / "sounds"
NLP_MODEL_PATH = DATA_DIR / "nlp_model"/"nlp_model.pkl"


# настройки
LANGUAGE = "ru-RU"
SAMPLE_RATE = 16000
LISTEN_TIME = 2.5

# модель Vosk
VOSK_MODEL_PATH = DATA_DIR / "vosk_model_small"

# Wake word
WAKE_WORD = DATA_DIR/"wakewords"/"Sebastian_en_windows_v3_0_0.ppn"

BEEP_WAKE = [
    SOUNDS_DIR / "What_do_you_order.mp3",
    SOUNDS_DIR / "Yes_Sir.mp3"
]
BEEP_ANSWER = [
    SOUNDS_DIR / "will_be_fulfilled.mp3",
    SOUNDS_DIR / "thats_right.mp3"
]
BEEP_START = [
    # SOUNDS_DIR / "FOR_THE_KING.mp3",
    SOUNDS_DIR / "Listen.mp3",
]
ACCESS_KEY = "ebG6PfQrBJjCcq3DFE2/nyaa9rpzhFqvS1WFh86FMszqlTK7rx+JYA=="

#NLP
FUZZY_THRESHOLD = 60