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
    return CONFIG.get(key) or os.getenv(key.upper()) or default
