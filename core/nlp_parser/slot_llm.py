from typing import List, Dict, Any, Optional
import json
import re
import time
import ast

try:
    import requests

    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

try:
    import llama_cpp

    _HAS_LLAMA_CPP = True
except Exception:
    _HAS_LLAMA_CPP = False


class VikhrSlotExtractor:
    def __init__(self,
                 backend: str = "ollama",
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "wavecut/vikhr",
                 llama_model_path: Optional[str] = None,
                 llama_kwargs: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.llama_model_path = llama_model_path
        self.llama_kwargs = llama_kwargs or {}
        self._llama = None

        if self.backend == "llama_cpp" and not _HAS_LLAMA_CPP:
            raise RuntimeError("llama_cpp backend requested but 'llama-cpp-python' not installed")
        if self.backend == "ollama" and not _HAS_REQUESTS:
            raise RuntimeError("requests library required for ollama backend")

    def _clean_response(self, text: str) -> str:
        """Удаляет markdown-обвязку (```json ... ```) если она есть."""
        text = text.strip()
        pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return text

    def _safe_json_extract(self, text: str) -> Optional[Dict[str, Any]]:
        # Сначала чистим от маркдауна
        clean_text = self._clean_response(text)

        start = clean_text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(clean_text)):
            char = clean_text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = clean_text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return ast.literal_eval(candidate)
                        except Exception:
                            return None
        return None

    def _coerce_types(self, slots_data: Dict[str, Any], slot_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Приводит значения к нужным типам (str -> int, str -> float)."""
        result = {}
        spec_map = {s["name"]: s.get("type", "str") for s in slot_specs}

        for key, value in slots_data.items():
            if key not in spec_map:
                continue  # Пропускаем галлюцинации (лишние ключи)

            target_type = spec_map[key]

            if value is None:
                result[key] = None
                continue

            try:
                if target_type == "int":
                    result[key] = int(float(str(value).replace(",", ".")))  # "5.0" -> 5
                elif target_type == "float":
                    result[key] = float(str(value).replace(",", "."))
                elif target_type == "bool":
                    s_val = str(value).lower()
                    result[key] = s_val in ["true", "1", "yes", "да", "t"]
                else:
                    result[key] = str(value)
            except Exception:
                # Если не смогли привести тип, оставляем как есть
                result[key] = value

        return result

    def _build_prompt(self, intent: str, text: str, slot_specs: List[Dict[str, Any]], example: Optional[str]) -> str:
        # Формируем строгую схему
        schema_desc = {}
        for s in slot_specs:
            schema_desc[s["name"]] = f"Type: {s.get('type', 'str')}, Required: {s.get('required', False)}"

        # Безопасное экранирование текста пользователя
        safe_text = json.dumps(text, ensure_ascii=False)
        safe_schema = json.dumps(schema_desc, ensure_ascii=False, indent=2)

        prompt = f"""### Instruction:
You are a smart NLU (Natural Language Understanding) system.
Analyze the User Command and extract arguments (slots) according to the Schema.

Intent: "{intent}"
Schema:
{safe_schema}

Rules:
1. Return ONLY a JSON object. No explanations.
2. If a required argument is missing, try to infer it from context. If impossible, use null.
3. If an optional argument is missing, use null.
4. Do NOT translate values unless necessary.

"""
        if example:
            prompt += f"Example Command: {example}\n"

        prompt += f"""
### User Command:
{safe_text}

### Response (JSON):
"""
        return prompt

    # ----- Backend Calls -----

    def _call_ollama(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        # Температура ниже (0.1) для большей стабильности
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "format": "json"  # Ollama поддерживает форсирование JSON
        }
        try:
            r = requests.post(url, json=payload, timeout=20)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def _init_llama(self):
        if self._llama is not None:
            return
        self._llama = llama_cpp.Llama(model_path=self.llama_model_path, **self.llama_kwargs)

    def _call_llama(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        self._init_llama()
        resp = self._llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["###", "User Command"]  # Стоп-слова чтобы не бредил
        )
        return resp.get("choices", [{}])[0].get("text", "")

    # ----- Main Method -----

    def extract_slots(self,
                      intent_name: str,
                      normalized_text: str,
                      slot_specs: List[Dict[str, Any]],
                      example: Optional[str] = None,
                      max_tokens: int = 256,
                      temperature: float = 0.1) -> Dict[str, Any]:

        prompt = self._build_prompt(intent_name, normalized_text, slot_specs, example)
        start = time.time()

        raw_response = ""
        try:
            if self.backend == "ollama":
                raw_response = self._call_ollama(prompt, max_tokens, temperature)
            elif self.backend == "llama_cpp":
                raw_response = self._call_llama(prompt, max_tokens, temperature)
            else:
                raise RuntimeError(f"Unknown backend {self.backend}")
        except Exception as e:
            return {"ok": False, "slots": {}, "raw": "", "explain": f"model error: {e}"}

        duration = time.time() - start

        # Парсим JSON
        parsed_data = self._safe_json_extract(raw_response)

        if parsed_data is None:
            # Fallback для одного числа (если модель тупая и вернула просто "5")
            if len(slot_specs) == 1:
                single_slot = slot_specs[0]
                name = single_slot["name"]
                # Ищем число
                m = re.search(r"-?\d+([.,]\d+)?", raw_response)
                if m:
                    val_str = m.group(0).replace(",", ".")
                    try:
                        val = float(val_str) if "." in val_str else int(val_str)
                        return {
                            "ok": True,
                            "slots": {name: val},
                            "raw": raw_response,
                            "explain": "fallback regex number",
                            "duration": duration
                        }
                    except:
                        pass

            return {
                "ok": False,
                "slots": {},
                "raw": raw_response,
                "explain": "failed to parse JSON",
                "duration": duration
            }

        # Приводим типы и фильтруем лишнее
        final_slots = self._coerce_types(parsed_data, slot_specs)

        return {
            "ok": True,
            "slots": final_slots,
            "raw": raw_response,
            "explain": "parsed json",
            "duration": duration
        }

# ---------------- Demo ----------------
if __name__ == "__main__":
    print("--- Инициализация LLM Extractor ---")
    ext = VikhrSlotExtractor(
        backend="ollama",
        model_name="llama3:8b",
        ollama_url="http://localhost:11434"
    )

    test_cases = [
        {
            "name": "Простой таймер",
            "intent": "set_timer",
            "text": "поставь таймер на 10 секунд",
            "slots": [{"name": "minutes", "type": "int"}, {"name": "seconds", "type": "int"}]
        },
        {
            "name": "Таймер текстом (проверка типов)",
            "intent": "set_timer",
            "text": "таймер на пять минут",
            "slots": [{"name": "minutes", "type": "int"}]
        },
        {
            "name": "Свет (Enum/String)",
            "intent": "turn_on_light",
            "text": "включи свет на кухне",
            "slots": [{"name": "location", "type": "str", "required": True}, {"name": "color", "type": "str"}]
        },
        {
            "name": "Неполная команда (проверка null)",
            "intent": "set_timer",
            "text": "таймер",
            "slots": [{"name": "minutes", "type": "int", "required": True}]
        }
    ]

    for case in test_cases:
        print(f"\n Тест: {case['name']}")
        print(f"   Вход: '{case['text']}'")

        res = ext.extract_slots(
            intent_name=case["intent"],
            normalized_text=case["text"],
            slot_specs=case["slots"]
        )

        if res["ok"]:
            print(f"   Успех: {res['slots']}")
        else:
            print(f"   Ошибка: {res['explain']}")
            print(f"   Raw: {res['raw']}")
