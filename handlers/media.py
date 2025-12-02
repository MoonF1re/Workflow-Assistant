import pyautogui
from core.logger import logger


class MediaController:
    def __init__(self):
        # Определяем OS для корректных горячих клавиш
        import platform
        self.os = platform.system()

    def _send_media_key(self, key):
        """Отправляет медиа-клавишу"""
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            logger.error(f"Ошибка отправки клавиши {key}: {e}")
            return False

    def media_play_pause(self):
        """Play/Pause"""
        # Пробел работает в большинстве медиаплееров
        self._send_media_key('playpause')
        return {"success": True, "message": "Воспроизведение приостановлено/возобновлено"}

    def media_next(self):
        """Следующий трек"""
        # Ctrl+Right или специальная медиа-клавиша
        self._send_media_key('nexttrack')
        return {"success": True, "message": "Следующий трек"}

    def media_previous(self):
        """Предыдущий трек"""
        # Ctrl+Left или специальная медиа-клавиша
        self._send_media_key('prevtrack')
        return {"success": True, "message": "Предыдущий трек"}

    def media_stop(self):
        """Стоп"""
        self._send_media_key('stop')
        return {"success": True, "message": "Воспроизведение остановлено"}


# Создаем экземпляр контроллера
media_controller = MediaController()


# Экспортируем функции
def media_play():
    return media_controller.media_play_pause()


def media_pause():
    return media_controller.media_play_pause()


def media_next():
    return media_controller.media_next()


def media_previous():
    return media_controller.media_previous()