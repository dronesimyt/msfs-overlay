import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_cfg

THEMES_DIR = Path(get_cfg("themes_dir", "./themes")).resolve()
THEMES_JSON = Path(get_cfg("themes_json", str(THEMES_DIR / "themes.json"))).resolve()

_theme_cache: Dict[str, Dict[str, Any]] = {}
_theme_mtime: Dict[str, float] = {}
_themes_json_cache: Dict[str, Any] = {}
_themes_json_mtime: Optional[float] = None


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_theme_for_icao(icao: str) -> Dict[str, Any]:
    icao = (icao or "default").upper()

    if THEMES_JSON.exists():
        global _themes_json_cache, _themes_json_mtime
        mtime = THEMES_JSON.stat().st_mtime
        if _themes_json_mtime != mtime or not _themes_json_cache:
            _themes_json_cache = _read_json(THEMES_JSON)
            _themes_json_mtime = mtime

        cfg = _themes_json_cache or {}
        default_cfg = cfg.get("default", {}) if isinstance(cfg, dict) else {}
        themes_map = cfg.get("themes", {}) if isinstance(cfg, dict) else {}

        chosen = themes_map.get(icao)
        if not isinstance(chosen, dict):
            icao = "default"
            chosen = default_cfg if isinstance(default_cfg, dict) else {}

        colors = chosen.get("colors") if isinstance(chosen.get("colors"), dict) else {}
        logo = chosen.get("logo")
        if isinstance(logo, dict):
            logo = logo.get("light") or logo.get("dark")

        return {
            "icao": icao,
            "name": chosen.get("name", icao),
            "colors": {
                "primary": colors.get("primary", "#ffffff"),
                "secondary": colors.get("secondary", "#999999"),
                "text": colors.get("text", "#ffffff"),
            },
            "logo": logo,
        }

    theme_dir = THEMES_DIR / icao
    theme_file = theme_dir / "theme.json"
    if not theme_file.exists():
        icao = "default"
        theme_dir = THEMES_DIR / "default"
        theme_file = theme_dir / "theme.json"

    if not theme_file.exists():
        return {
            "icao": "default", "name": "Default",
            "colors": {"primary": "#ffffff", "secondary": "#999999", "text": "#ffffff"},
            "logo": None,
        }

    mtime = theme_file.stat().st_mtime
    cache_key = str(theme_file)
    if _theme_cache.get(cache_key) is not None and _theme_mtime.get(cache_key) == mtime:
        return _theme_cache[cache_key]

    theme = _read_json(theme_file)
    result = {
        "icao": icao,
        "name": theme.get("name", icao),
        "colors": theme.get("colors", {
            "primary": theme.get("primary", "#ffffff"),
            "secondary": theme.get("secondary", "#999999"),
            "text": theme.get("text", "#ffffff"),
        }),
        "logo": theme.get("logo"),
    }
    _theme_cache[cache_key] = result
    _theme_mtime[cache_key] = mtime
    return result
