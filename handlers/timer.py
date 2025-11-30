def set_timer(minutes: int = 0, seconds: int = 0):
    # Защита от None (превращаем в 0)
    m = minutes or 0
    s = seconds or 0

    total_seconds = m * 60 + s

    if total_seconds == 0:
        return "Вы не указали время для таймера."

    print(f"Запускаю таймер на {total_seconds} сек...")