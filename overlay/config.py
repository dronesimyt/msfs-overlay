import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


CONFIG = load_config()


def get_cfg(key: str, default: Any = None) -> Any:
    """config.json first, then the upper-case env var, then the default.

    Only missing/empty values fall through, so 0 or false are respected.
    """
    v = CONFIG.get(key)
    if v is not None and v != "":
        return v
    v = os.getenv(key.upper())
    if v is not None and v != "":
        return v
    return default


def get_cfg_int(key: str, default: int) -> int:
    try:
        return int(get_cfg(key, default))
    except (TypeError, ValueError):
        return default
