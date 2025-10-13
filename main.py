from core.recognizer import Recognizer

def main():
    recog = Recognizer()

    try:
        recog.run()
    except KeyboardInterrupt:
        recog.stop()

if __name__ == "__main__":
    main()

