from importlib import import_module
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import inspect
import time
import traceback
from typing import Dict, Any, List, Optional

def _coerce_value(val: Any, target_type: Optional[str]):
    if val is None:
        return None
    if not target_type:
        return val
    t = str(target_type).lower()
    try:
        if t in ("int", "integer"):
            return int(val)
        if t in ("float", "double"):
            return float(val)
        if t in ("bool", "boolean"):
            if isinstance(val, bool):
                return val
            s = str(val).strip().lower()
            if s in ("true", "1", "yes", "y", "да", "вкл"):
                return True
            if s in ("false", "0", "no", "n", "нет", "выкл"):
                return False
            return bool(val)
        # default -> str
        return str(val)
    except Exception:
        # возврат в исходном виде при неудаче
        return val

class Dispatcher:
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _import_handler(self, handler_ref: str):
        if not handler_ref or not isinstance(handler_ref, str):
            return None, f"invalid handler_ref: {handler_ref}"

        ref = handler_ref.strip()

        if ":" in ref:
            mod_name, func_name = ref.split(":", 1)
        else:
            parts = ref.split(".")
            if len(parts) < 2:
                return None, f"invalid handler_ref format: {ref}"
            mod_name = ".".join(parts[:-1])
            func_name = parts[-1]

        try:
            mod = import_module(mod_name)
        except Exception as e:
            return None, f"failed to import module {mod_name}: {e}"

        func = getattr(mod, func_name, None)
        if func is None:
            return None, f"module {mod_name} has no attribute {func_name}"
        if not callable(func):
            return None, f"{handler_ref} is not callable"
        return func, None

    def _prepare_call_args(self, func, slots: Dict[str, Any], slot_specs: Optional[List[Dict[str, Any]]]):

        sig = inspect.signature(func)
        params = sig.parameters  # ordered dict
        args_to_pass = {}
        missing = []

        spec_map = {}
        if slot_specs:
            for s in slot_specs:
                if isinstance(s, dict) and s.get("name"):
                    spec_map[s["name"]] = s

        for pname, p in params.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            if pname in slots:
                raw_val = slots[pname]
                expected_type = spec_map.get(pname, {}).get("type") if spec_map else None
                coerced = _coerce_value(raw_val, expected_type)
                args_to_pass[pname] = coerced
            else:
                if p.default is inspect._empty:
                    missing.append(pname)
                else:
                    pass

        return args_to_pass, missing

    def _call_sync(self, func, kwargs: Dict[str, Any]):
        try:
            res = func(**kwargs)
            return res, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"{e}\n{tb}"

    def call(self, handler_ref: str, slots: Dict[str, Any], slot_specs: Optional[List[Dict[str, Any]]] = None,
             timeout: float = 5.0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        func, err = self._import_handler(handler_ref)
        if func is None:
            return {"ok": False, "status": "handler_not_found", "result": None, "error": err, "used_args": {}, "missing": [], "duration": time.perf_counter() - t0}

        args_kwargs, missing = self._prepare_call_args(func, slots or {}, slot_specs)
        if missing:
            duration = time.perf_counter() - t0
            return {"ok": False, "status": "missing_slots", "result": None, "error": f"missing required slots: {missing}", "used_args": args_kwargs, "missing": missing, "duration": duration}

        future = self._executor.submit(self._call_sync, func, args_kwargs)
        try:
            res, errstr = future.result(timeout=timeout)
            duration = time.perf_counter() - t0
            if errstr:
                return {"ok": False, "status": "error", "result": None, "error": errstr, "used_args": args_kwargs, "missing": [], "duration": duration}
            return {"ok": True, "status": "ok", "result": res, "error": None, "used_args": args_kwargs, "missing": [], "duration": duration}
        except TimeoutError:
            duration = time.perf_counter() - t0
            return {"ok": False, "status": "timeout", "result": None, "error": f"handler timed out after {timeout}s", "used_args": args_kwargs, "missing": [], "duration": duration}
        except Exception as e:
            duration = time.perf_counter() - t0
            return {"ok": False, "status": "error", "result": None, "error": str(e), "used_args": args_kwargs, "missing": [], "duration": duration}
