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

def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    try:
                        return ast.literal_eval(candidate)
                    except Exception:
                        return None
    return None

_PROMPT_TEMPLATE = """Ты — интеллектуальный ассистент. 
Твоя задача — извлечь значения слотов (аргументов) из пользовательской команды.

Формат ответа:
- Верни **только JSON-объект** (без комментариев и текста вне JSON).
- Если аргумент не указан — поставь `null`.
- Используй правильные типы данных (int, float, bool, str).
- Не переводи слоты и их аргументы

Данные:
Intent: {intent}
Команда: "{text}"
Ожидаемые слоты: {slots_list}

Примеры:
{{"minutes": 5, "seconds": null}}

Теперь проанализируй команду выше и выдай JSON с заполненными слотами. **Не переводя их на русский**.
Ответ должен содержать **только JSON**.
"""


class VikhrSlotExtractor:
    def __init__(self,
                 backend: str = "ollama",
                 ollama_url: str = "http://localhost:11434",
                 llama_model_path: Optional[str] = None,
                 llama_kwargs: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.ollama_url = ollama_url.rstrip("/")
        self.llama_model_path = llama_model_path
        self.llama_kwargs = llama_kwargs or {}
        self._llama = None
        if self.backend == "llama_cpp" and not _HAS_LLAMA_CPP:
            raise RuntimeError("llama_cpp backend requested but 'llama-cpp-python' not installed")
        if self.backend == "ollama" and not _HAS_REQUESTS:
            raise RuntimeError("requests library required for ollama backend")

    def _build_prompt(self, intent: str, text: str, slot_specs: List[Dict[str, Any]], example: Optional[str]) -> str:
        slot_list = []
        for s in slot_specs:
            name = s.get("name")
            t = s.get("type", "str")
            req = s.get("required", False)
            slot_list.append(f"{name} (type={t}, required={req})")
        slot_list_str = "[" + ", ".join(slot_list) + "]"
        prompt = _PROMPT_TEMPLATE.format(intent=intent, text=text.replace('"', '\\"'), slots_list=slot_list_str)
        if example:
            prompt += f"\n# matched example: {example}\n"
        return prompt

    # ----- Ollama HTTP -----
    def _call_ollama(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": "wavecut/vikhr",
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()

            if isinstance(data, dict):
                return data.get("response", "")

            return str(data)

        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def _init_llama(self):
        if self._llama is not None:
            return
        self._llama = llama_cpp.Llama(model_path=self.llama_model_path, **self.llama_kwargs)

    def _call_llama(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        self._init_llama()
        resp = self._llama.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        if isinstance(resp, dict):
            text = resp.get("content") or resp.get("text") or ""
            if not text:
                choices = resp.get("choices", [])
                if choices and isinstance(choices, list):
                    text = choices[0].get("text", "") or choices[0].get("message", {}).get("content", "")
            return text or str(resp)
        return str(resp)

    def extract_slots(self,
                      intent_name: str,
                      normalized_text: str,
                      slot_specs: List[Dict[str, Any]],
                      example: Optional[str] = None,
                      max_tokens: int = 256,
                      temperature: float = 0.0,
                      timeout: float = 15.0) -> Dict[str, Any]:

        prompt = self._build_prompt(intent_name, normalized_text, slot_specs, example)
        start = time.time()
        try:
            if self.backend == "ollama":
                resp_text = self._call_ollama(prompt, max_tokens=max_tokens, temperature=temperature)
            elif self.backend == "llama_cpp":
                resp_text = self._call_llama(prompt, max_tokens=max_tokens, temperature=temperature)
            else:
                raise RuntimeError(f"Unknown backend {self.backend}")
        except Exception as e:
            return {"ok": False, "slots": {}, "raw": "", "explain": f"model call failed: {e}"}

        duration = time.time() - start
        parsed = _safe_json_extract(resp_text)
        if parsed is None:
            if len(slot_specs) == 1:
                single = slot_specs[0]
                name = single.get("name")
                m = re.search(r"-?\d+[\,\.]?\d*", resp_text)
                if m:
                    sval = m.group(0).replace(",", ".")
                    try:
                        if "." in sval:
                            val = float(sval)
                        else:
                            val = int(sval)
                        return {"ok": True, "slots": {name: val}, "raw": resp_text, "explain": "parsed simple number", "duration": duration}
                    except Exception:
                        pass
            return {"ok": False, "slots": {}, "raw": resp_text, "explain": "could not parse JSON from model response", "duration": duration}

        expected = [s.get("name") for s in slot_specs]
        out_slots = {}
        for k, v in parsed.items():
            if k in expected:
                out_slots[k] = v
        return {"ok": True, "slots": out_slots, "raw": resp_text, "explain": "parsed json", "duration": duration}

# ---------------- Demo ----------------
if __name__ == "__main__":
    print("Demo VikhrSlotExtractor")
    ext = VikhrSlotExtractor(backend="ollama", ollama_url="http://localhost:11434")

    result = ext.extract_slots(
        intent_name="set_timer",
        normalized_text="поставь таймер на пять минут",
        slot_specs=[{"name": "minutes", "type": "int", "required": True}],
        example="поставь таймер на {minutes} минут"
    )

    print(result)
    # If you have a GGUF file and llama-cpp-python:
    # ext2 = VikhrSlotExtractor(backend="llama_cpp", llama_model_path="/path/to/vikhr.gguf")
    # print(ext2.extract_slots(intent, "поставь таймер на 5 минут", slots, example))
