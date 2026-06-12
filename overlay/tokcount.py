import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

import requests

from .config import CONFIG

TIKTOK_USER_ID = CONFIG.get("tiktok_user_id")
TOKCOUNT_REFRESH_SECONDS = int(CONFIG.get("tokcount_refresh_seconds", 15))
_CACHE_PATH = Path(__file__).parent.parent / "tokcount_cache.json"

_state: Dict[str, Any] = {
    "followers": None, "likes": None, "following": None,
    "videos": None, "ts": 0.0, "error": None,
}
_lock = threading.Lock()

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


def _save_cache():
    try:
        tmp = str(_CACHE_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: _state[k] for k in ("ts", "followers", "likes", "following", "videos")}, f, indent=2)
        os.replace(tmp, str(_CACHE_PATH))
    except Exception:
        pass


def get_tokcount() -> Dict[str, Any]:
    if not TIKTOK_USER_ID:
        return {**_state, "error": "tiktok_user_id not set"}

    now = time.time()
    with _lock:
        if _state["followers"] is not None and (now - _state["ts"]) < TOKCOUNT_REFRESH_SECONDS:
            return dict(_state)

    try:
        data = _fetch(TIKTOK_USER_ID)
        if not data.get("success"):
            raise RuntimeError(f"tokcount success=false: {data}")
        with _lock:
            _state.update({
                "ts": now,
                "followers": int(data["followerCount"]) if data.get("followerCount") is not None else None,
                "likes": int(data["likeCount"]) if data.get("likeCount") is not None else None,
                "following": int(data["followingCount"]) if data.get("followingCount") is not None else None,
                "videos": int(data["videoCount"]) if data.get("videoCount") is not None else None,
                "error": None,
            })
            _save_cache()
        return dict(_state)
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
        return dict(_state)


def _worker():
    while True:
        try:
            get_tokcount()
        except Exception:
            pass
        time.sleep(max(5, TOKCOUNT_REFRESH_SECONDS))


threading.Thread(target=_worker, daemon=True).start()
