"""
Recognizer -> Normalizer -> Registry -> IntentClassifier -> LLM -> Dispatcher
"""

import time
import json
import os
from pathlib import Path

from core.recognizer import Recognizer
from core.nlp_parser.normalizer import Normalizer
from core.nlp_parser.registry import Registry
from core.nlp_parser.intent_classifier import IntentClassifier
from core.nlp_parser.slot_resolver import SlotResolver
from core.nlp_parser.slot_llm import VikhrSlotExtractor
from core.dispatcher import Dispatcher
from core.config import NLP_MODEL_PATH, COMMANDS_DIR
from core.logger import logger

MODEL_PATH = Path(NLP_MODEL_PATH)


# ----- Подготовка подсистем -----
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
            logger.info(f"[main] Классификатор загружен из {model_path}")
            return clf
        except Exception as e:
            logger.warning(f"[main] Не удалось загрузить классификатор из {model_path}: {e} — будем тренировать заново")

    texts, labels = reg.build_intent_dataset(normalizer=normalizer)
    if not texts:
        raise RuntimeError("Нет примеров для обучения. Добавьте файлы examples/<intent>.txt")
    clf.fit(texts, labels)

    try:
        # попытка сохранить модель
        model_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save(str(model_path))
        logger.info(f"[main] Обученный классификатор сохранён в {model_path}")
    except Exception as e:
        logger.warning(f"[main] Не удалось сохранить классификатор в {model_path}: {e}")
    return clf


def prepare_llm_extractor(backend: str = "ollama", backend_kwargs: dict = None) -> VikhrSlotExtractor:
    backend_kwargs = backend_kwargs or {}
    if backend == "ollama":
        ollama_url = backend_kwargs.get("ollama_url", "http://localhost:11434")
        return VikhrSlotExtractor(backend="ollama", ollama_url=ollama_url, model_name='llama3:8b')
    elif backend == "llama_cpp":
        model_path = backend_kwargs.get("model_path")
        if not model_path:
            raise RuntimeError("llama_cpp backend requires 'model_path' in backend_kwargs")
        return VikhrSlotExtractor(backend="llama_cpp", llama_model_path=model_path, llama_kwargs=backend_kwargs.get("llama_kwargs", {}))
    else:
        raise RuntimeError(f"Неподдерживаемый LLM backend: {backend}")


# ----- Создание обработчика команд (вызывается Recognizer) -----
def make_on_command(normalizer: Normalizer, registry: Registry, classifier: IntentClassifier,
                    llm_extractor: VikhrSlotExtractor, dispatcher: Dispatcher):
    slot_resolver = SlotResolver(registry)

    def on_command(raw_text: str):
        t0 = time.perf_counter()
        try:
            # 1) Нормализация
            norm = normalizer.normalize(raw_text)
            normalized = norm.get("normalized", "") if isinstance(norm, dict) else str(norm)

            # 2) Предсказание интента
            try:
                intent_label, prob = classifier.predict(normalized)
            except Exception:
                # очень простой fallback: если predict падает, считаем неизвестно
                intent_label, prob = None, 0.0

            # Вывод минимальной информации
            print(f'Вход: "{raw_text}"')
            print(f'-> нормализовано: "{normalized}"')
            if intent_label:
                print(f'-> интент: {intent_label} (p={prob:.2f})')
            else:
                print('-> интент: <не распознан>')

            # 3) Извлекаем слоты
            cmd_spec = registry.find_command(intent_label)
            slot_specs = registry.get_slot_spec(intent_label)

            final_slots = {}

            if slot_specs:
                # --- ЭТАП 3.1: SlotResolver (Regex) ---
                print("-> Запуск SlotResolver...")
                regex_slots = slot_resolver.resolve(intent_label, normalized)
                if regex_slots:
                    print(f"   Regex нашел: {regex_slots}")

                final_slots.update(regex_slots)

                # Проверяем, каких обязательных слотов не хватает
                required_names = {s['name'] for s in slot_specs if s.get('required')}
                found_names = set(final_slots.keys())
                missing_required = required_names - found_names

                # --- ЭТАП 3.2: LLM (только если чего-то не хватает или вообще нет hints) ---
                # Можно настроить логику: если hints есть, но они не сработали -> зовем LLM
                # Или если hints покрыли все required -> LLM не зовем (экономия времени!)

                regex_missed_everything = (len(slot_specs) > 0 and not final_slots)

                if missing_required or regex_missed_everything:
                    print(f"-> Не хватает обязательных слотов {missing_required}, вызываю LLM...")
                    # Передаем пример из registry, если есть
                    example_text = cmd_spec.examples[0] if cmd_spec.examples else None

                    llm_res = llm_extractor.extract_slots(intent_label, normalized, slot_specs, example=example_text)

                    llm_slots = llm_res.get("slots", {}) if isinstance(llm_res, dict) else {}
                    print(f"   LLM вернул: {llm_slots}")

                    # Объединяем.
                    # СТРАТЕГИЯ: Regex (надежный) перезаписывает LLM (креативный), если ключи совпадают?
                    # Или наоборот? Обычно Regex точнее.
                    # Берем всё из LLM, а потом обновляем значениями из Regex
                    merged = llm_slots.copy()
                    merged.update(regex_slots)  # Приоритет у Regex
                    final_slots = merged
                else:
                    print("-> Все обязательные слоты найдены через Regex. LLM пропущен.")

            print(f"Итоговые слоты: {final_slots}")

            # 4) Если есть handler — вызываем через dispatcher
            cmd_spec = registry.find_command(intent_label) if intent_label else None
            handler_ref = getattr(cmd_spec, "handler", None) if cmd_spec else None
            if handler_ref:
                # передаём dispatcher найденные слоты и спецификацию слотов
                res = dispatcher.call(handler_ref, final_slots, slot_specs=slot_specs, timeout=6.0)
                if res["ok"]:
                    print(f"-> обработчик вернул: {res['result']}")
                else:
                    print(f"-> ошибка при вызове обработчика: {res['status']}; {res.get('error')}")
            else:
                if intent_label:
                    print("-> обработчик для этого интента не указан (handler отсутствует)")
                # если интент не распознан — ничего не вызываем

            total = time.perf_counter() - t0
            print("======")
            print(f"Время: {total:.2f} сек.\n")
        except Exception as e:
            logger.exception(f"[main.on_command] Ошибка при обработке команды: {e}")
    return on_command


# ----- Микрофонный режим -----
def run_interactive(backend="ollama", backend_kwargs=None):
    normalizer = Normalizer(convert_numbers=True, lemmatize=False)
    registry = prepare_registry()
    classifier = prepare_intent_classifier(registry, normalizer, MODEL_PATH)
    llm_extractor = prepare_llm_extractor(backend=backend, backend_kwargs=backend_kwargs)
    dispatcher = Dispatcher()

    on_command = make_on_command(normalizer, registry, classifier, llm_extractor, dispatcher)
    recog = Recognizer(on_command=on_command)
    try:
        recog.run()
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl-C")
        recog.stop()
    except Exception as e:
        logger.exception(f"[main] Непредвиденная ошибка: {e}")
        recog.stop()


# ----- Режим без микрофона (Откладка) -----
def run_stdin_test(backend="ollama", backend_kwargs=None):
    normalizer = Normalizer(convert_numbers=True, lemmatize=False)
    registry = prepare_registry()
    classifier = prepare_intent_classifier(registry, normalizer, MODEL_PATH)
    llm_extractor = prepare_llm_extractor(backend=backend, backend_kwargs=backend_kwargs)
    dispatcher = Dispatcher()

    on_command = make_on_command(normalizer, registry, classifier, llm_extractor, dispatcher)

    print("Режим ввода (stdin). Введите фразу и нажмите Enter (пустая строка — выход).")
    while True:
        try:
            s = input("> ").strip()
        except EOFError:
            break
        if not s:
            break
        on_command(s)


# ----- Точка входа -----
if __name__ == "__main__":
    MODE = os.environ.get("ASSISTANT_MODE", "mic")  # "mic" или "stdin"
    LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
    LLM_KW = {}
    if LLM_BACKEND == "llama_cpp":
        mpath = os.environ.get("LLM_MODEL_PATH")
        if not mpath:
            raise RuntimeError("LLM_BACKEND=llama_cpp требует указать LLM_MODEL_PATH (путь к GGUF модели)")
        LLM_KW["model_path"] = mpath

    if MODE == "mic":
        run_interactive(backend=LLM_BACKEND, backend_kwargs=LLM_KW)
    else:
        run_stdin_test(backend=LLM_BACKEND, backend_kwargs=LLM_KW)
