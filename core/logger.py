import logging
from pathlib import Path

# создаём директорию для логов
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "assistant.log"

# базовая настройка логгера
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("assistant")


# ---- ПРИМЕР ИСПОЛЬЗОВАНИЯ -----
# from core.logger import logger

# logger.info("Команда выполнена")

# --- ВЫВОД ----
# [2025-10-12 19:25:01,680] INFO: Команда выполнена