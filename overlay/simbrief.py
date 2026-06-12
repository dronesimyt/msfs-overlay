import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import CONFIG
from .nav import to_float

SIMBRIEF_CACHE_SECONDS = int(CONFIG.get("simbrief_cache_seconds", 60))

_simbrief_cache = {"ts": 0.0, "data": None, "error": None}
_simbrief_lock = threading.Lock()
_session = requests.Session()


# -----------------------------
# Utilities
# -----------------------------

def deep_get(d: Any, *path: str) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def json_safe(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore").strip("\x00").strip()
        except Exception:
            return str(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    return str(v)


# -----------------------------
# ICAO helpers
# -----------------------------

def normalize_icao(code: Optional[str], max_len: int = 4) -> Optional[str]:
    if not code:
        return None
    c = "".join(ch for ch in str(code).upper().strip() if ch.isalnum())
    return c[:max_len] if len(c) >= 3 else None


def infer_icao_from_callsign(callsign: Optional[str]) -> Optional[str]:
    if not callsign:
        return None
    cs = "".join(ch for ch in str(callsign).upper().strip() if ch.isalnum())
    return cs[:3] if len(cs) >= 3 and cs[:3].isalpha() else None


def parse_aircraft_icao_from_fpl_text(fpl_text: Optional[str]) -> Optional[str]:
    if not fpl_text:
        return None
    m = re.search(r"\n-([A-Z0-9]{3,4})\/", str(fpl_text))
    return normalize_icao(m.group(1), max_len=4) if m else None


def normalize_aircraft_icao(v: Any) -> Optional[str]:
    s = json_safe(v)
    if s is None:
        return None
    s = str(s).strip().upper()
    if s in {"", "NONE", "NULL", "N/A", "TEXT", "STRING"}:
        return None
    m = re.search(r"\b([A-Z0-9]{3,4})\b", s)
    return normalize_icao(m.group(1), max_len=4) if m else None


# -----------------------------
# Fetch + cache
# -----------------------------

def _get_identity() -> Tuple[Optional[str], Optional[str]]:
    username = CONFIG.get("simbrief_username") or os.getenv("SIMBRIEF_USERNAME")
    userid = CONFIG.get("simbrief_userid") or os.getenv("SIMBRIEF_USERID")
    return username, userid


def _fetch_latest():
    params = {"json": "1"}
    username, userid = _get_identity()
    if userid:
        params["userid"] = userid
    elif username:
        params["username"] = username
    else:
        return None, "SIMBRIEF_USERNAME or SIMBRIEF_USERID not set"
    try:
        r = _session.get("https://www.simbrief.com/api/xml.fetcher.php", params=params, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def get_simbrief_cached():
    now = time.time()
    with _simbrief_lock:
        if _simbrief_cache["data"] is not None and (now - _simbrief_cache["ts"]) < SIMBRIEF_CACHE_SECONDS:
            return _simbrief_cache["data"], _simbrief_cache["error"]

    data, err = _fetch_latest()
    with _simbrief_lock:
        _simbrief_cache.update({"ts": now, "data": data, "error": err})
    return data, err


# -----------------------------
# Field extraction
# -----------------------------

def extract_simbrief_fields(sb: Optional[dict]) -> Dict[str, Any]:
    if not sb:
        return {}
    out: Dict[str, Any] = {
        "callsign": (
            deep_get(sb, "atc", "callsign")
            or deep_get(sb, "general", "callsign")
            or deep_get(sb, "general", "flight_number")
        ),
        "dep_icao": (
            deep_get(sb, "origin", "icao_code") or deep_get(sb, "origin", "icao")
            or deep_get(sb, "origin", "icao_id") or deep_get(sb, "origin", "id")
        ),
        "arr_icao": (
            deep_get(sb, "destination", "icao_code") or deep_get(sb, "destination", "icao")
            or deep_get(sb, "destination", "icao_id") or deep_get(sb, "destination", "id")
        ),
        "airline_icao": (
            deep_get(sb, "general", "icao_airline") or deep_get(sb, "airline", "icao_code")
            or deep_get(sb, "airline", "icao") or deep_get(sb, "atc", "airline_icao")
        ),
        "aircraft_icao": (
            deep_get(sb, "aircraft", "icao_code") or deep_get(sb, "aircraft", "icao")
            or deep_get(sb, "general", "icao_type")
            or parse_aircraft_icao_from_fpl_text(deep_get(sb, "atc", "flightplan_text"))
        ),
    }
    out["dep_icao"] = normalize_icao(out.get("dep_icao"), max_len=4)
    out["arr_icao"] = normalize_icao(out.get("arr_icao"), max_len=4)
    out["airline_icao"] = normalize_icao(out.get("airline_icao"), max_len=3)
    out["aircraft_icao"] = normalize_icao(out.get("aircraft_icao"), max_len=4)
    return {k: json_safe(v) for k, v in out.items() if v is not None}


def resolve_airline_icao(sb_fields: Dict[str, Any]) -> Tuple[str, str]:
    sb_icao = normalize_icao(sb_fields.get("airline_icao"), max_len=3)
    if sb_icao:
        return sb_icao, "simbrief"
    cs_icao = infer_icao_from_callsign(sb_fields.get("callsign"))
    if cs_icao:
        return cs_icao, "callsign"
    return "default", "default"


def build_route_points_from_simbrief(sb_raw: Optional[dict]) -> List[Tuple[float, float]]:
    if not sb_raw:
        return []
    points: List[Tuple[float, float]] = []

    o_lat = to_float(deep_get(sb_raw, "origin", "pos_lat"))
    o_lon = to_float(deep_get(sb_raw, "origin", "pos_long"))
    if o_lat is not None and o_lon is not None:
        points.append((o_lat, o_lon))

    for wp in (sb_raw.get("navlog") or []):
        if not isinstance(wp, dict):
            continue
        wlat = to_float(wp.get("pos_lat") or wp.get("lat"))
        wlon = to_float(wp.get("pos_long") or wp.get("lon"))
        if wlat is not None and wlon is not None:
            points.append((wlat, wlon))

    d_lat = to_float(deep_get(sb_raw, "destination", "pos_lat"))
    d_lon = to_float(deep_get(sb_raw, "destination", "pos_long"))
    if d_lat is not None and d_lon is not None:
        points.append((d_lat, d_lon))

    cleaned: List[Tuple[float, float]] = []
    for p in points:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    return cleaned
