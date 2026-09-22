"""TikTok follower stats via tokcount.com.

TikTok has no public API for follower counts, so this uses the unofficial
endpoint behind tokcount.com. It can break or rate-limit at any time, so:
- the last known values are persisted and restored on startup,
- failures back off exponentially instead of hammering the endpoint,
- values older than STALE_SECONDS are flagged as stale,
- set "tokcount_enabled": false in config.json to turn it off completely.
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

import requests

from .config import get_cfg, get_cfg_int

TIKTOK_USER_ID = get_cfg("tiktok_user_id")
TOKCOUNT_ENABLED = str(get_cfg("tokcount_enabled", True)).lower() not in {"0", "false", "no", "off"}
TOKCOUNT_REFRESH_SECONDS = max(5, get_cfg_int("tokcount_refresh_seconds", 15))
TOKCOUNT_MAX_BACKOFF_SECONDS = 15 * 60
STALE_SECONDS = 30 * 60
_CACHE_PATH = Path(__file__).parent.parent / "tokcount_cache.json"
_STAT_KEYS = ("followers", "likes", "following", "videos")

_state: Dict[str, Any] = {
    "followers": None, "likes": None, "following": None,
    "videos": None, "ts": 0.0, "error": None,
}
_lock = threading.Lock()
_failures = 0

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Referer": "https://tokcount.com/",
    "Origin": "https://tokcount.com",
    "Connection": "keep-alive",
})


def _fetch(uid: str) -> dict:
    r = _session.get(f"https://tiktok.tokcount.com/user/stats/{uid}", timeout=10)
    r.raise_for_status()
    return r.json()


def _to_int(v: Any) -> Any:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _load_cache():
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        _state.update({k: _to_int(data.get(k)) for k in _STAT_KEYS})
        _state["ts"] = float(data.get("ts") or 0.0)
    except Exception:
        pass


def _save_cache():
    try:
        tmp = str(_CACHE_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: _state[k] for k in ("ts", *_STAT_KEYS)}, f, indent=2)
        os.replace(tmp, str(_CACHE_PATH))
    except Exception:
        pass


def _snapshot() -> Dict[str, Any]:
    snap = dict(_state)
    snap["stale"] = snap["followers"] is not None and (time.time() - snap["ts"]) > STALE_SECONDS
    return snap


def _disabled_reason() -> Any:
    if not TOKCOUNT_ENABLED:
        return "tokcount disabled"
    if not TIKTOK_USER_ID:
        return "tiktok_user_id not set"
    return None


def get_tokcount_cached() -> Dict[str, Any]:
    """Returns the last known stats without network access (refreshed by the worker)."""
    reason = _disabled_reason()
    if reason:
        return {**_state, "stale": False, "error": reason}
    with _lock:
        return _snapshot()


def get_tokcount() -> Dict[str, Any]:
    global _failures
    reason = _disabled_reason()
    if reason:
        return {**_state, "stale": False, "error": reason}

    now = time.time()
    try:
        data = _fetch(TIKTOK_USER_ID)
        if not data.get("success"):
            raise RuntimeError(f"tokcount success=false: {data}")
        with _lock:
            _state.update({
                "ts": now,
                "followers": _to_int(data.get("followerCount")),
                "likes": _to_int(data.get("likeCount")),
                "following": _to_int(data.get("followingCount")),
                "videos": _to_int(data.get("videoCount")),
                "error": None,
            })
            _failures = 0
            _save_cache()
            return _snapshot()
    except Exception as e:
        with _lock:
            _failures += 1
            _state["error"] = str(e)
            return _snapshot()


def _next_delay() -> float:
    if _failures == 0:
        return TOKCOUNT_REFRESH_SECONDS
    return min(TOKCOUNT_MAX_BACKOFF_SECONDS, TOKCOUNT_REFRESH_SECONDS * (2 ** min(_failures, 10)))


def _worker():
    while True:
        try:
            get_tokcount()
        except Exception:
            pass
        time.sleep(_next_delay())


_load_cache()
if _disabled_reason() is None:
    threading.Thread(target=_worker, daemon=True).start()
