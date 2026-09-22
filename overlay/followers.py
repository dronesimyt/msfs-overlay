"""TikTok follower stats.

TikTok has no public API for follower counts, so the numbers come from
unofficial sources that can break at any time. Two providers are tried in
order:

1. "tiktok_web"  - the public profile page, which embeds the stats as JSON.
                   Needs "tiktok_username" (the @name without the @).
2. "tokcount"    - tokcount.com's endpoint. Needs "tiktok_user_id".

A background worker refreshes the value, failures back off exponentially,
the last known value is cached on disk and restored on startup, and values
older than STALE_SECONDS are flagged as stale.
"""
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from .config import get_cfg, get_cfg_int

TIKTOK_USERNAME = str(get_cfg("tiktok_username", "") or "").lstrip("@")
TIKTOK_USER_ID = get_cfg("tiktok_user_id")
FOLLOWERS_ENABLED = str(get_cfg("followers_enabled", True)).lower() not in {"0", "false", "no", "off"}
# The profile page is scraped, so don't hammer it: once a minute is plenty.
REFRESH_SECONDS = max(15, get_cfg_int("followers_refresh_seconds", 60))
MAX_BACKOFF_SECONDS = 15 * 60
STALE_SECONDS = 30 * 60

_CACHE_PATH = Path(__file__).parent.parent / "followers_cache.json"
_STAT_KEYS = ("followers", "likes", "following", "videos")

_state: Dict[str, Any] = {
    "followers": None, "likes": None, "following": None, "videos": None,
    "ts": 0.0, "source": None, "error": None,
}
_lock = threading.Lock()
_failures = 0

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_session = requests.Session()
_session.headers.update({
    "User-Agent": _UA,
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
})

_REHYDRATION_RE = re.compile(
    r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', re.S
)
_FOLLOWER_RE = re.compile(r'"followerCount":\s*"?(\d+)"?')


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _fetch_tiktok_web() -> Dict[str, Any]:
    """Reads the stats JSON that TikTok embeds in the public profile page."""
    r = _session.get(
        f"https://www.tiktok.com/@{TIKTOK_USERNAME}",
        headers={"Accept": "text/html,application/xhtml+xml"},
        timeout=15,
    )
    r.raise_for_status()

    stats: Dict[str, Any] = {}
    m = _REHYDRATION_RE.search(r.text)
    if m:
        try:
            data = json.loads(m.group(1))
            info = data["__DEFAULT_SCOPE__"]["webapp.user-detail"]["userInfo"]
            stats = info.get("stats") or info.get("statsV2") or {}
        except Exception:
            stats = {}

    if not stats:
        # Fall back to a plain search in case TikTok reshuffles the page.
        m = _FOLLOWER_RE.search(r.text)
        if not m:
            raise RuntimeError("no follower count in profile page (blocked or layout changed?)")
        stats = {"followerCount": m.group(1)}

    followers = _to_int(stats.get("followerCount"))
    if followers is None:
        raise RuntimeError("profile page returned no followerCount")

    return {
        "followers": followers,
        "likes": _to_int(stats.get("heartCount") or stats.get("heart")),
        "following": _to_int(stats.get("followingCount")),
        "videos": _to_int(stats.get("videoCount")),
    }


def _fetch_tokcount() -> Dict[str, Any]:
    r = _session.get(
        f"https://tiktok.tokcount.com/user/stats/{TIKTOK_USER_ID}",
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://tokcount.com/",
            "Origin": "https://tokcount.com",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"tokcount success=false: {data}")
    return {
        "followers": _to_int(data.get("followerCount")),
        "likes": _to_int(data.get("likeCount")),
        "following": _to_int(data.get("followingCount")),
        "videos": _to_int(data.get("videoCount")),
    }


def _providers() -> list:
    out = []
    if TIKTOK_USERNAME:
        out.append(("tiktok_web", _fetch_tiktok_web))
    if TIKTOK_USER_ID:
        out.append(("tokcount", _fetch_tokcount))
    return out


def _disabled_reason() -> Optional[str]:
    if not FOLLOWERS_ENABLED:
        return "followers disabled"
    if not _providers():
        return "tiktok_username or tiktok_user_id not set"
    return None


def _load_cache():
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        _state.update({k: _to_int(data.get(k)) for k in _STAT_KEYS})
        _state["ts"] = float(data.get("ts") or 0.0)
        _state["source"] = data.get("source")
    except Exception:
        pass


def _save_cache():
    try:
        tmp = str(_CACHE_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: _state[k] for k in ("ts", "source", *_STAT_KEYS)}, f, indent=2)
        os.replace(tmp, str(_CACHE_PATH))
    except Exception:
        pass


def _snapshot() -> Dict[str, Any]:
    snap = dict(_state)
    snap["stale"] = snap["followers"] is not None and (time.time() - snap["ts"]) > STALE_SECONDS
    return snap


def get_followers_cached() -> Dict[str, Any]:
    """Last known stats without network access (the worker keeps them fresh)."""
    reason = _disabled_reason()
    if reason:
        return {**_state, "stale": False, "error": reason}
    with _lock:
        return _snapshot()


def refresh() -> Dict[str, Any]:
    """Tries every provider in order and stores the first result that works."""
    global _failures
    reason = _disabled_reason()
    if reason:
        return {**_state, "stale": False, "error": reason}

    errors = []
    for name, fetch in _providers():
        try:
            stats = fetch()
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        with _lock:
            _state.update(stats)
            _state.update({"ts": time.time(), "source": name, "error": None})
            _failures = 0
            _save_cache()
            return _snapshot()

    with _lock:
        _failures += 1
        _state["error"] = "; ".join(errors)
        return _snapshot()


def _next_delay() -> float:
    if _failures == 0:
        return REFRESH_SECONDS
    return min(MAX_BACKOFF_SECONDS, REFRESH_SECONDS * (2 ** min(_failures, 10)))


def _worker():
    while True:
        try:
            refresh()
        except Exception:
            pass
        time.sleep(_next_delay())


_load_cache()
if _disabled_reason() is None:
    threading.Thread(target=_worker, daemon=True).start()
