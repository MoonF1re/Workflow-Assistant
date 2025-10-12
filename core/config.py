from pathlib import Path

# базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# пути
DATA_DIR = BASE_DIR / "data"
COMMANDS_DIR = BASE_DIR / "commands"
LOGS_DIR = BASE_DIR / "logs"

# настройки
LANGUAGE = "ru-RU"
SAMPLE_RATE = 16000

# модель Vosk — укажем позже путь после загрузки
VOSK_MODEL_PATH = DATA_DIR / "vosk_model"
