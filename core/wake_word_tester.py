import struct
import time
import pvporcupine
import pyaudio
from playsound import playsound
from config import ACCESS_KEY, WAKE_WORD, BEEP_WAKE


def play_beep():
    try:
        playsound(str(BEEP_WAKE[1]), block=False)
    except Exception as e:
        print(f"Ошибка воспроизведения звука: {e}")

def main():


    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths= [WAKE_WORD]
    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Скажи - Себастьян!!")

    while True:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        start = time.perf_counter()
        result = porcupine.process(pcm)
        end = time.perf_counter()

        latency = (end - start) * 1000  # в миллисекундах

        if result >= 0:
            play_beep()
            print(f"Активировано! Время реакции: {latency:.2f} мс")

if __name__ == "__main__":
    main()
