from core.logger import logger
from core import config

def main():
    logger.info("Ассистент запущен.")
    logger.info(f"Рабочая директория: {config.BASE_DIR}")

if __name__ == "__main__":
    main()
