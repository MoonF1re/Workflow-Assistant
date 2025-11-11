import json
import struct
import time
import pyaudio
import random
import pvporcupine
from vosk import Model, KaldiRecognizer
from playsound import playsound
from queue import Queue
from core.config import WAKE_WORD, VOSK_MODEL_PATH, SAMPLE_RATE, BEEP_ANSWER, BEEP_WAKE, BEEP_START, ACCESS_KEY, LISTEN_TIME
from core.logger import logger

def play_random_start():
    """Проигрывает случайный звук запуска"""
    sound_path = random.choice(BEEP_START)
    playsound(str(sound_path), block=False)

def play_random_wake():
    """Проигрывает случайный звук активации."""
    sound_path = random.choice(BEEP_WAKE)
    playsound(str(sound_path), block=False)

def play_random_answer():
    """Проигрывает случайный звук ответа"""
    sound_path = random.choice(BEEP_ANSWER)
    playsound(str(sound_path), block=False)


class Recognizer:
    def __init__(self,  on_command=None):
        logger.info("Инициализация распознавателя...")

        # === Wake Word ===
        self.porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=[str(WAKE_WORD)]
        )

        # === Vosk ===
        self.model = Model(str(VOSK_MODEL_PATH))
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)

        # === Audio ===
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        self.stream.start_stream()

        self.on_command = on_command

        self.command_queue = Queue()
        self.running = True

        logger.info("Распознаватель инициализирован")
        play_random_start()


    # --- Основной цикл ---
    def run(self):
        logger.info("Ожидание wake word...")

        while self.running:
            pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            result = self.porcupine.process(pcm_unpacked)

            if result >= 0:
                logger.info("Wake word активировано!")
                play_random_wake()

                command_text = self.listen_command()
                if command_text:
                    self.command_queue.put(command_text)
                    logger.info(f"Распознано: {command_text}")

                    if self.on_command:
                        self.on_command(command_text)

    # --- Прослушивание команды ---
    def listen_command(self, silence_timeout=LISTEN_TIME):
        logger.info("Слушаю команду...")
        self.recognizer.Reset()

        last_speech_time = time.time()
        collected_text = ""

        while True:
            data = self.stream.read(4000, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    collected_text += " " + text
                    last_speech_time = time.time()
            else:
                # обновляем, если есть частичные данные
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
                if partial:
                    last_speech_time = time.time()

            # проверяем тишину
            if time.time() - last_speech_time > silence_timeout:
                final = json.loads(self.recognizer.FinalResult()).get("text", "")
                collected_text += " " + final
                collected_text = collected_text.strip()
                if collected_text:
                    logger.info(f"Команда: {collected_text}")
                    play_random_answer()
                    return collected_text
                else:
                    logger.info("Команда не распознана")
                    return None

    def get_command(self):
        if not self.command_queue.empty():
            return self.command_queue.get()
        return None

    def stop(self):
        logger.info("Остановка распознавателя...")
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.porcupine.delete()
        logger.info("Ресурсы освобождены.")
