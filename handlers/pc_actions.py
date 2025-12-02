import webbrowser
import pyautogui
import time
from pathlib import Path
from datetime import datetime

# Словарь коротких имен для удобства
SHORTCUTS = {
    "вк": "https://vk.com",
    "vk": "https://vk.com",
    "Ютуб": "https://youtube.com",
    "youtube": "https://youtube.com",
    "гугл": "https://google.com",
    "google": "https://google.com",
    "яндекс": "https://yandex.ru",
    "yandex": "https://yandex.ru",
    "гитхаб": "https://github.com",
    "github": "https://github.com"
}


def open_website(site_name: str = None):
    """
    Открывает сайт в браузере по умолчанию.
    """
    if not site_name:
        webbrowser.open("https://google.com")
        return "Открываю браузер."

    target = site_name.lower().strip()

    # 1. Проверяем в словаре коротких имен
    if target in SHORTCUTS:
        url = SHORTCUTS[target]
    # 2. Если похоже на домен (есть точка), оставляем как есть
    elif "." in target:
        if not target.startswith("http"):
            url = f"https://{target}"
        else:
            url = target
    # 3. Иначе ищем в гугле
    else:
        url = f"https://www.google.com/search?q={target}"

    webbrowser.open(url)
    return f"Открываю {target}"


def take_screenshot(delay: int = 0):
    """
    Делает скриншот и сохраняет в папку Screenshots.
    """
    save_dir = Path("data/screenshots")
    save_dir.mkdir(parents=True, exist_ok=True)

    if delay is None:
        delay = 0

    if delay > 0:
        time.sleep(delay)

    # Генерируем имя файла: screenshot_2023-10-25_14-30-01.png
    filename = f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    filepath = save_dir / filename

    try:
        pyautogui.screenshot(filepath)
        return f"Скриншот сохранён: {filename}"
    except Exception as e:
        return f"Ошибка при создании скриншота: {e}"

def write_text():
    print("Это функция выводит текст")