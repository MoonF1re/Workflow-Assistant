import time
import json
import os
from pathlib import Path

from core.recognizer import Recognizer
from core.nlp_parser.normalizer import Normalizer
from core.nlp_parser.registry import Registry
from core.nlp_parser.intent_classifier import IntentClassifier
from core.nlp_parser.slot_llm import VikhrSlotExtractor
from core.config import NLP_MODEL_PATH, COMMANDS_DIR
from core.logger import logger

MODEL_PATH = Path(NLP_MODEL_PATH)

# Параметры быстрого обучения (если модели нет)
TRAIN_TEST_SPLIT = 0.15

# ----- Инициализация подсистем -----
def prepare_registry(commands_dir: str = None, examples_dir: str = None) -> Registry:
    commands_dir = commands_dir or COMMANDS_DIR
    examples_dir = examples_dir or (Path(commands_dir).parent / "examples")
    reg = Registry(commands_dir=str(commands_dir), examples_dir=str(examples_dir))
    reg.load()
    return reg

def prepare_intent_classifier(reg: Registry, normalizer: Normalizer, model_path: Path) -> IntentClassifier:
    clf = IntentClassifier(use_transformer=True)
    if model_path.exists():
        try:
            clf.load(str(model_path))
            logger.info(f"[main] Loaded intent classifier from {model_path}")
            return clf
        except Exception as e:
            logger.warning(f"[main] Failed to load classifier from {model_path}: {e} — будем тренировать заново")


    texts, labels = reg.build_intent_dataset(normalizer=normalizer)
    if not texts:
        raise RuntimeError("No training examples found in registry. Add examples in examples/*.txt first.")
    clf.fit(texts, labels)
    # сохраняем
    try:
        clf.save(str(model_path))
        logger.info(f"[main] Trained and saved intent classifier to {model_path}")
    except Exception as e:
        logger.warning(f"[main] Trained classifier but failed to save to {model_path}: {e}")
    return clf

def prepare_llm_extractor(backend: str = "ollama", backend_kwargs: dict = None) -> VikhrSlotExtractor:
    backend_kwargs = backend_kwargs or {}
    if backend == "ollama":
        ollama_url = backend_kwargs.get("ollama_url", "http://localhost:11434")
        ext = VikhrSlotExtractor(backend="ollama", ollama_url=ollama_url)
    elif backend == "llama_cpp":
        model_path = backend_kwargs.get("model_path")
        if not model_path:
            raise RuntimeError("llama_cpp backend requires 'model_path' in backend_kwargs")
        ext = VikhrSlotExtractor(backend="llama_cpp", llama_model_path=model_path, llama_kwargs=backend_kwargs.get("llama_kwargs", {}))
    else:
        raise RuntimeError(f"Unsupported LLM backend: {backend}")
    return ext

def make_on_command(normalizer: Normalizer, registry: Registry, classifier: IntentClassifier, llm_extractor: VikhrSlotExtractor):
    def on_command(raw_text: str):
        t0 = time.perf_counter()
        try:
            # Normalize
            norm = normalizer.normalize(raw_text)
            normalized = norm.get("normalized", "") if isinstance(norm, dict) else str(norm)
            t1 = time.perf_counter()

            # Intent prediction
            try:
                intent_label, prob = classifier.predict(normalized)
            except Exception:
                # fallback: attempt to call predict_proba then top label
                try:
                    pmap = classifier.predict_proba([normalized])[0]
                    if pmap:
                        # get best
                        intent_label, prob = max(pmap.items(), key=lambda kv: kv[1])
                    else:
                        intent_label, prob = None, 0.0
                except Exception:
                    intent_label, prob = None, 0.0
            t2 = time.perf_counter()

            # Print minimal header
            print(f'Input: "{raw_text}"')
            print(f'-> normalized: "{normalized}"')
            if intent_label:
                print(f'-> intent: {intent_label} (p={prob:.2f})')
            else:
                print("-> intent: <unknown>")

            # If there are slots for this intent, call LLM to extract
            slot_spec = registry.get_slot_spec(intent_label) if intent_label else None
            if slot_spec:
                print("-> calling LLM to extract slots...")
                t_llm_start = time.perf_counter()
                llm_res = llm_extractor.extract_slots(intent_label, normalized, slot_spec, example=None)
                t_llm_end = time.perf_counter()
                # attempt to pretty-print slots or raw
                slots = llm_res.get("slots") if isinstance(llm_res, dict) else None
                raw_llm = llm_res.get("raw") if isinstance(llm_res, dict) else str(llm_res)
                if slots:
                    print(f"LLM returned: {json.dumps(slots, ensure_ascii=False)}")
                else:
                    # nothing parsed
                    print(f"LLM returned: {raw_llm!r}")
                t3 = time.perf_counter()
            else:
                # no slots expected -> nothing to do
                t_llm_end = t2
                t3 = t2

            total = time.perf_counter() - t0
            print("======")
            print(f"Время: {total:.2f} сек.")
            print("", flush=True)
        except Exception as e:
            logger.exception(f"[main.on_command] Error processing command: {e}")
    return on_command

# ----- Main runner (connect Recognizer) -----
def run_interactive(backend="ollama", backend_kwargs=None):
    # prepare components
    normalizer = Normalizer(convert_numbers=True, lemmatize=False)
    registry = prepare_registry()
    classifier = prepare_intent_classifier(registry, normalizer, MODEL_PATH)
    llm_extractor = prepare_llm_extractor(backend=backend, backend_kwargs=backend_kwargs)

    # create on_command and start recognizer
    on_command = make_on_command(normalizer, registry, classifier, llm_extractor)
    recog = Recognizer(on_command=on_command)
    try:
        recog.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        recog.stop()
    except Exception as e:
        logger.exception(f"[main] Unexpected error: {e}")
        recog.stop()

def run_stdin_test(backend="ollama", backend_kwargs=None):
    normalizer = Normalizer(convert_numbers=True, lemmatize=False)
    registry = prepare_registry()
    classifier = prepare_intent_classifier(registry, normalizer, MODEL_PATH)
    llm_extractor = prepare_llm_extractor(backend=backend, backend_kwargs=backend_kwargs)

    on_command = make_on_command(normalizer, registry, classifier, llm_extractor)

    print("Stdin test mode. Type a phrase and press Enter (empty line to exit).")
    while True:
        try:
            s = input("> ").strip()
        except EOFError:
            break
        if not s:
            break
        on_command(s)

# ----- Entrypoint -----
if __name__ == "__main__":
    # Choose which mode: 'mic' to use recognizer, 'stdin' for manual testing
    MODE = os.environ.get("ASSISTANT_MODE", "mic")  # set ASSISTANT_MODE=mic to use microphone
    # Optionally set backend via env var, e.g. LLM_BACKEND=llama_cpp and LLM_MODEL_PATH
    LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
    LLM_KW = {}
    if LLM_BACKEND == "llama_cpp":
        mpath = os.environ.get("LLM_MODEL_PATH")
        if not mpath:
            raise RuntimeError("LLM_BACKEND=llama_cpp requires env LLM_MODEL_PATH pointing to GGUF model")
        LLM_KW["model_path"] = mpath

    if MODE == "mic":
        run_interactive(backend=LLM_BACKEND, backend_kwargs=LLM_KW)
    else:
        run_stdin_test(backend=LLM_BACKEND, backend_kwargs=LLM_KW)


#
# import json
# from core.recognizer import Recognizer
# from core.logger import logger
# from core.config import COMMANDS_DIR
# from core.nlp_parser_old.intent_manager import IntentManager
#
# # --- 1. Инициализация IntentManager ---
# im = IntentManager(COMMANDS_DIR)
#
#
# # --- 2. Обработчик распознанной команды ---
# def handle_command(text: str):
#     """Обрабатывает распознанную голосовую команду через IntentManager"""
#     if not text or not text.strip():
#         logger.warning("Пустая команда, игнорируем.")
#         return
#
#     logger.info(f"Распознано: {text}")
#
#     result = im.parse(text)
#     logger.debug(f"Результат анализа: {json.dumps(result, ensure_ascii=False, indent=2)}")
#
#     intent = result.get("intent_name")
#     slots = result.get("slots", {})
#     confidence = result.get("confidence", 0.0)
#     match_type = result.get("match_type", "none")
#
#     if not intent:
#         print(f"Команда не распознана (match_type={match_type}, confidence={confidence:.1f})")
#         return
#
#     print(f"Найден intent: {intent} (match_type={match_type}, confidence={confidence:.1f}%)")
#     if slots:
#         print(f"   → slots: {slots}")
#     else:
#         print("   → без параметров")
#
#     # --- Пример обработки команды ---
#     if intent == "set_timer":
#         minutes = slots.get("minutes", 0)
#         print(f"Устанавливаю таймер на {minutes} минут(ы)")
#
#     else:
#         print(f"Выполняю действие для intent '{intent}'")
#
#
# # --- 3. Главная функция ---
# def main():
#     recog = Recognizer(on_command=handle_command)
#
#     try:
#         recog.run()
#     except KeyboardInterrupt:
#         logger.info("Прерывание пользователем")
#         recog.stop()
#
#
# if __name__ == "__main__":
#     main()
