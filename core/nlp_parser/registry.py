from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import yaml
import json



@dataclass
class SlotSpec:
    # Описание параметров команды
    name: str
    type: str = "str"
    required: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SlotSpec":
        return SlotSpec(
            name=d.get("name"),
            type=d.get("type", "str"),
            required=bool(d.get("required", False))
        )

@dataclass
class CommandSpec:
    # Описание команды
    name: str
    handler: Optional[str] = None
    slots: List[SlotSpec] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    filepath: Optional[Path] = None


class Registry:
    def __init__(self, commands_dir: str = "commands", examples_dir: str = "examples"):
        self.commands_dir = Path(commands_dir)
        self.examples_dir = Path(examples_dir)
        self.commands: Dict[str, CommandSpec] = {}

    # ---------- Loading helpers ----------
    def _load_examples_for(self, name: str) -> List[str]:
        # Загружает примеры фраз для команды из файлов
        out: List[str] = []
        p_txt = self.examples_dir / f"{name}.txt"
        p_yml = self.examples_dir / f"{name}.yml"
        if p_txt.exists():
            try:
                with p_txt.open("r", encoding="utf-8") as f:
                    for ln in f:
                        s = ln.strip()
                        if s:
                            out.append(s)
            except Exception:
                pass
        elif p_yml.exists():
            try:
                with p_yml.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    if isinstance(data, list):
                        out = [str(x).strip() for x in data if str(x).strip()]
                    elif isinstance(data, dict) and "examples" in data and isinstance(data["examples"], list):
                        out = [str(x).strip() for x in data["examples"] if str(x).strip()]
            except Exception:
                pass
        return out

    def _validate_card(self, data: Dict[str, Any], path: Path) -> Optional[str]:
        # Проверяет, что данные команды корректны.
        if not isinstance(data, dict):
            return f"{path}: not a mapping"
        name = data.get("name")
        if not name:
            return f"{path}: missing required field 'name'"
        return None

    def load(self) -> None:
        # Загружает все команды из папки commands_dir и связывает их с примерами
        self.commands = {}
        if not self.commands_dir.exists():
            return

        for p in sorted(self.commands_dir.glob("*.yml")):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                # skip unreadable file
                print(f"[registry] failed to read {p}: {e}")
                continue

            err = self._validate_card(data, p)
            if err:
                print(f"[registry] validation error: {err}")
                continue

            name = str(data.get("name")).strip()
            handler = data.get("handler")
            slots_in = data.get("slots", []) or []
            slots = []
            for s in slots_in:
                if isinstance(s, dict) and s.get("name"):
                    slots.append(SlotSpec.from_dict(s))

            # load examples from examples/<name>.txt or .yml
            examples = self._load_examples_for(name)

            spec = CommandSpec(
                name=name,
                handler=handler,
                slots=slots,
                examples=examples,
                filepath=p
            )
            if name in self.commands:
                print(f"[registry] warning: duplicate command name {name} (overwriting)")
            self.commands[name] = spec

    # ---------- Public ----------
    def list_commands(self) -> List[str]:
        return list(self.commands.keys())

    def find_command(self, name: str) -> Optional[CommandSpec]:
        return self.commands.get(name)

    def get_slot_spec(self, intent_name: str) -> Optional[List[Dict[str, Any]]]:
        c = self.find_command(intent_name)
        if not c:
            return None
        return [{"name": s.name, "type": s.type, "required": s.required} for s in c.slots]

    def get_examples_for_intent(self, intent_name: str) -> List[str]:
        c = self.find_command(intent_name)
        if not c:
            return []
        return c.examples.copy()

    def build_intent_dataset(self, normalizer: Optional[Any] = None) -> Tuple[List[str], List[str]]:
        # Создает данные для обучения intent_classifier
        texts: List[str] = []
        labels: List[str] = []
        for name, spec in self.commands.items():
            for ex in spec.examples:
                if normalizer is not None:
                    try:
                        nr = normalizer.normalize(ex)
                        txt = nr.get("normalized", "") if isinstance(nr, dict) else str(nr)
                    except Exception:
                        txt = ex
                else:
                    txt = ex
                if txt:
                    texts.append(txt)
                    labels.append(name)
        return texts, labels

    def build_fuzzy_index(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for name, spec in self.commands.items():
            for ex in spec.examples:
                out.append((name, ex))
        return out

    def export_dataset_jsonl(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for name, spec in self.commands.items():
                for ex in spec.examples:
                    obj = {"text": ex, "intent": name}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------- Demo ----------------
if __name__ == "__main__":
    # Demo expects a directory structure:
    # commands/  (with minimal YAMLs)
    # examples/  (with matching <name>.txt files)
    print("Registry demo (loads commands/ and examples/ relative to cwd)")
    reg = Registry(commands_dir="commands", examples_dir="examples")
    reg.load()
    print("Loaded commands:", reg.list_commands())

    # Show per-command summary
    for name in reg.list_commands():
        spec = reg.find_command(name)
        print("---")
        print("name:", spec.name)
        print("handler:", spec.handler)
        print("slots:", [{"name":s.name,"type":s.type,"required":s.required} for s in spec.slots])
        print("examples count:", len(spec.examples))
        if spec.examples:
            print("example[0]:", spec.examples[0])

    texts, labels = reg.build_intent_dataset(normalizer=None)
    print("\nDataset size:", len(texts))
    if texts:
        print("sample:", texts[:5], labels[:5])

    reg.export_dataset_jsonl("out/intent_dataset.jsonl")
    print("Exported dataset to out/intent_dataset.jsonl")
