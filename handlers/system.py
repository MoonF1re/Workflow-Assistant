import os
import subprocess
import pyautogui
import time
import ctypes
from core.logger import logger


class SystemController:
    def __init__(self):
        self.os = os.name

    def open_calculator(self):
        """Открывает калькулятор"""
        try:
            if self.os == 'nt':  # Windows
                os.system('calc')
            elif self.os == 'posix':  # Linux/Mac
                os.system('gnome-calculator')
            return {"success": True, "message": "Калькулятор открыт"}
        except Exception as e:
            return {"success": False, "message": f"Ошибка: {e}"}

    def open_notepad(self):
        """Открывает блокнот"""
        try:
            if self.os == 'nt':
                os.system('notepad')
            elif self.os == 'posix':
                os.system('gedit')
            return {"success": True, "message": "Блокнот открыт"}
        except Exception as e:
            return {"success": False, "message": f"Ошибка: {e}"}

    def lock_computer(self):
        """Блокирует компьютер"""
        try:
            if self.os == 'nt':
                ctypes.windll.user32.LockWorkStation()
                return {"success": True, "message": "Компьютер заблокирован"}
            elif self.os == 'posix':
                os.system('gnome-screensaver-command -l')
                return {"success": True, "message": "Компьютер заблокирован"}
        except Exception as e:
            return {"success": False, "message": f"Ошибка: {e}"}

    def shutdown(self, seconds=30):
        """Выключение компьютера"""
        try:
            if self.os == 'nt':
                os.system(f'shutdown /s /t {seconds}')
                return {"success": True, "message": f"Компьютер выключится через {seconds} секунд"}
            elif self.os == 'posix':
                os.system(f'shutdown -h +{seconds // 60}')
                return {"success": True, "message": f"Компьютер выключится через {seconds} секунд"}
        except Exception as e:
            return {"success": False, "message": f"Ошибка: {e}"}

    def cancel_shutdown(self):
        """Отменяет выключение"""
        try:
            if self.os == 'nt':
                os.system('shutdown /a')
                return {"success": True, "message": "Выключение отменено"}
            elif self.os == 'posix':
                os.system('shutdown -c')
                return {"success": True, "message": "Выключение отменено"}
        except Exception as e:
            return {"success": False, "message": f"Ошибка: {e}"}


# Создаем экземпляр контроллера
system_controller = SystemController()


# Экспортируем функции
def volume_up():
    """Увеличивает громкость"""
    try:
        # 15 нажатий клавиши увеличения громкости
        for _ in range(15):
            pyautogui.press('volumeup')
        return {"success": True, "message": "Громкость увеличена"}
    except Exception as e:
        return {"success": False, "message": f"Ошибка: {e}"}


def volume_down():
    """Уменьшает громкость"""
    try:
        # 15 нажатий клавиши уменьшения громкости
        for _ in range(15):
            pyautogui.press('volumedown')
        return {"success": True, "message": "Громкость уменьшена"}
    except Exception as e:
        return {"success": False, "message": f"Ошибка: {e}"}


def volume_mute():
    """Включает/выключает звук"""
    try:
        pyautogui.press('volumemute')
        return {"success": True, "message": "Звук переключен"}
    except Exception as e:
        return {"success": False, "message": f"Ошибка: {e}"}


def open_calculator():
    """Открывает калькулятор"""
    return system_controller.open_calculator()


def open_notepad():
    """Открывает блокнот"""
    return system_controller.open_notepad()


def lock_computer():
    """Блокирует компьютер"""
    return system_controller.lock_computer()


def shutdown(seconds: int = 30):
    """Выключает компьютер через указанное время"""
    return system_controller.shutdown(seconds)


def cancel_shutdown():
    """Отменяет выключение"""
    return system_controller.cancel_shutdown()