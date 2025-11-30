import re
from typing import Dict, Any, Optional, List
from core.nlp_parser.registry import Registry


class SlotResolver:
    def __init__(self, registry: Registry):
        self.registry = registry

    def _cast_value(self, value: str, target_type: str) -> Any:
        """Простое приведение типов для извлеченных строк"""
        try:
            if target_type == "int":
                return int(value)
            elif target_type == "float":
                return float(value.replace(",", "."))
            elif target_type == "bool":
                return value.lower() in ("true", "1", "yes", "да")
            return value
        except Exception:
            return value

    def resolve(self, intent_name: str, text: str) -> Dict[str, Any]:
        found_slots = {}
        cmd_spec = self.registry.find_command(intent_name)

        slot_types = {s.name: s.type for s in cmd_spec.slots}

        for slot_name, hints in cmd_spec.slot_hints.items():
            patterns = hints.get("patterns", [])
            target_type = slot_types.get(slot_name, "str")

            for pattern in patterns:
                # --- ОТЛАДКА: смотрим какой паттерн проверяем ---
                # print(f"[SlotResolver] Проверяю паттерн: '{pattern}' для слота '{slot_name}' на тексте '{text}'")

                try:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        if slot_name in match.groupdict():
                            raw_val = match.group(slot_name)
                            # print(f"[SlotResolver] MATCH! {slot_name} = {raw_val}")
                            found_slots[slot_name] = self._cast_value(raw_val, target_type)
                            break
                        else:
                            # fallback для групп без имени
                            raw_val = match.group(1) if match.groups() else match.group(0)
                            found_slots[slot_name] = self._cast_value(raw_val, target_type)
                            break
                except re.error as e:
                    print(f"[SlotResolver] Ошибка в регулярке {slot_name}: {pattern} ({e})")
                    continue

        return found_slots