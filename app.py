import math
import os
import time
import json
import threading
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, jsonify, send_file, send_from_directory

from SimConnect import SimConnect, AircraftRequests

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
CONFIG_PATH = Path(__file__).with_name("config.json")


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


# Themes directory (user-editable)
THEMES_DIR = Path(get_cfg("themes_dir", "./themes")).resolve()


# SimBrief identity (config.json > env vars)
def get_simbrief_identity() -> Tuple[Optional[str], Optional[str]]:
    username = CONFIG.get("simbrief_username") or os.getenv("SIMBRIEF_USERNAME")
    userid = CONFIG.get("simbrief_userid") or os.getenv("SIMBRIEF_USERID")
    return username, userid


# -----------------------------
# SimBrief cache
# -----------------------------
SIMBRIEF_CACHE_SECONDS = int(get_cfg("simbrief_cache_seconds", 60))
_simbrief_cache = {"ts": 0.0, "data": None, "error": None}
_simbrief_lock = threading.Lock()


def _fetch_simbrief_latest():
    base = "https://www.simbrief.com/api/xml.fetcher.php"
    params = {"json": "1"}

    username, userid = get_simbrief_identity()
    if userid:
        params["userid"] = userid
    elif username:
        params["username"] = username
    else:
        return None, "SIMBRIEF_USERNAME or SIMBRIEF_USERID not set"

    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def get_simbrief_cached():
    now = time.time()
    with _simbrief_lock:
        age = now - _simbrief_cache["ts"]
        if _simbrief_cache["data"] is not None and age < SIMBRIEF_CACHE_SECONDS:
            return _simbrief_cache["data"], _simbrief_cache["error"]

    data, err = _fetch_simbrief_latest()
    with _simbrief_lock:
        _simbrief_cache["ts"] = now
        _simbrief_cache["data"] = data
        _simbrief_cache["error"] = err
    return data, err


def deep_get(d: Any, *path: str) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def json_safe(v: Any) -> Any:
    # bytes -> string
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore").strip("\x00").strip()
        except Exception:
            return str(v)

    # passthrough primitives
    if v is None or isinstance(v, (str, int, float, bool)):
        return v

    # nested structures
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [json_safe(x) for x in v]

    # last resort
    return str(v)


# -----------------------------
# SimBrief extraction
# -----------------------------
def normalize_icao(code: Optional[str], max_len: int = 4) -> Optional[str]:
    if not code:
        return None
    c = "".join(ch for ch in str(code).upper().strip() if ch.isalnum())
    if len(c) < 3:
        return None
    return c[:max_len]


def infer_icao_from_callsign(callsign: Optional[str]) -> Optional[str]:
    if not callsign:
        return None
    cs = "".join(ch for ch in str(callsign).upper().strip() if ch.isalnum())
    if len(cs) >= 3 and cs[:3].isalpha():
        return cs[:3]
    return None


def parse_aircraft_icao_from_fpl_text(fpl_text: Optional[str]) -> Optional[str]:
    """
    Example:
      (FPL-DLH444-IS
      -C172/L-SFG/C
      -EDKB1420
      ...)
    We want: C172
    """
    if not fpl_text:
        return None
    s = str(fpl_text)
    m = re.search(r"\n-([A-Z0-9]{3,4})\/", s)
    if not m:
        return None
    return normalize_icao(m.group(1), max_len=4)


def extract_simbrief_fields(sb: Optional[dict]) -> Dict[str, Any]:
    if not sb:
        return {}

    out: Dict[str, Any] = {}

    out["callsign"] = (
        deep_get(sb, "atc", "callsign")
        or deep_get(sb, "general", "callsign")
        or deep_get(sb, "general", "flight_number")
    )

    out["dep_icao"] = (
        deep_get(sb, "origin", "icao_code")
        or deep_get(sb, "origin", "icao")
        or deep_get(sb, "origin", "icao_id")
        or deep_get(sb, "origin", "id")
    )

    out["arr_icao"] = (
        deep_get(sb, "destination", "icao_code")
        or deep_get(sb, "destination", "icao")
        or deep_get(sb, "destination", "icao_id")
        or deep_get(sb, "destination", "id")
    )

    out["airline_icao"] = (
        deep_get(sb, "general", "icao_airline")
        or deep_get(sb, "airline", "icao_code")
        or deep_get(sb, "airline", "icao")
        or deep_get(sb, "atc", "airline_icao")
    )

    # Aircraft ICAO (best effort)
    out["aircraft_icao"] = (
        deep_get(sb, "aircraft", "icao_code")
        or deep_get(sb, "aircraft", "icao")
        or deep_get(sb, "general", "icao_type")
        or parse_aircraft_icao_from_fpl_text(deep_get(sb, "atc", "flightplan_text"))
    )

    # Normalize ICAO shapes
    out["dep_icao"] = normalize_icao(out.get("dep_icao"), max_len=4)
    out["arr_icao"] = normalize_icao(out.get("arr_icao"), max_len=4)
    out["airline_icao"] = normalize_icao(out.get("airline_icao"), max_len=3)
    out["aircraft_icao"] = normalize_icao(out.get("aircraft_icao"), max_len=4)

    return {k: json_safe(v) for k, v in out.items() if v is not None}


def resolve_airline_icao(sb_fields: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (icao_or_default, source)
    source in: simbrief | callsign | default
    """
    sb_icao = normalize_icao(sb_fields.get("airline_icao"), max_len=3)
    if sb_icao:
        return sb_icao, "simbrief"

    cs_icao = infer_icao_from_callsign(sb_fields.get("callsign"))
    if cs_icao:
        return cs_icao, "callsign"

    return "default", "default"


# -----------------------------
# Theme loading + cache
# -----------------------------
_theme_cache: Dict[str, Dict[str, Any]] = {}
_theme_mtime: Dict[str, float] = {}


def load_theme_for_icao(icao: str) -> Dict[str, Any]:
    """
    Loads themes/<ICAO>/theme.json. Falls back to themes/default/theme.json.
    Caches by mtime.
    """
    icao = (icao or "default").upper()
    theme_dir = THEMES_DIR / icao
    theme_file = theme_dir / "theme.json"

    if not theme_file.exists():
        icao = "default"
        theme_dir = THEMES_DIR / "default"
        theme_file = theme_dir / "theme.json"

    if not theme_file.exists():
        return {
            "name": "Default",
            "icao": "default",
            "colors": {"primary": "#ffffff", "secondary": "#999999", "text": "#ffffff"},
            "logo": None,
        }

    mtime = theme_file.stat().st_mtime
    cache_key = str(theme_file)

    if _theme_cache.get(cache_key) is not None and _theme_mtime.get(cache_key) == mtime:
        return _theme_cache[cache_key]

    try:
        theme = json.loads(theme_file.read_text(encoding="utf-8"))
    except Exception:
        theme = {}

    theme_out = {
        "icao": icao,
        "name": theme.get("name", icao),
        "colors": theme.get(
            "colors",
            {
                "primary": theme.get("primary", "#ffffff"),
                "secondary": theme.get("secondary", "#999999"),
                "text": theme.get("text", "#ffffff"),
            },
        ),
        "logo": theme.get("logo"),
    }

    _theme_cache[cache_key] = theme_out
    _theme_mtime[cache_key] = mtime
    return theme_out


# -----------------------------
# SimConnect (resilient)
# -----------------------------
sm = None
aq = None
_last_simconnect_state: Optional[bool] = None
_last_simconnect_msg_ts = 0.0


def log_once_on_state_change(ok: bool, msg: str):
    global _last_simconnect_state, _last_simconnect_msg_ts
    now = time.time()
    if _last_simconnect_state != ok or (now - _last_simconnect_msg_ts) > 15:
        print(msg)
        _last_simconnect_state = ok
        _last_simconnect_msg_ts = now


def ensure_connection() -> Tuple[Optional[AircraftRequests], bool, Optional[str]]:
    """
    Returns (aq_obj_or_none, ok, message)
    """
    global sm, aq
    try:
        if sm is None or aq is None:
            sm = SimConnect()
            aq = AircraftRequests(sm, _time=1000)  # 1s cache (matches your polling)
        log_once_on_state_change(True, "[SimConnect] Connected.")
        return aq, True, None
    except Exception as e:
        sm = None
        aq = None
        msg = f"[SimConnect] Not available (is MSFS running?). {e}"
        log_once_on_state_change(False, msg)
        return None, False, msg


def safe_get(aq_obj: Optional[AircraftRequests], name: str, default: Any = None) -> Any:
    if aq_obj is None:
        return default
    try:
        v = aq_obj.get(name)
        return default if v is None else v
    except Exception:
        return default


def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rad_to_deg(v: Any) -> Optional[float]:
    fv = to_float(v)
    if fv is None:
        return None
    return math.degrees(fv) % 360.0


# -----------------------------
# Distance / time helpers
# -----------------------------
def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Earth radius in NM
    r_nm = 3440.065
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_nm * c


def ll_to_nm_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    # local projection, in NM (good enough for route DTG)
    x = (lon - ref_lon) * math.cos(math.radians(ref_lat)) * 60.0
    y = (lat - ref_lat) * 60.0
    return x, y


def project_point_to_segment_nm(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float, float]:
    # returns (t in [0..1], closest_x, closest_y, dist_to_segment)
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return 0.0, ax, ay, math.hypot(px - ax, py - ay)

    t = (wx * vx + wy * vy) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * vx, ay + t * vy
    d = math.hypot(px - cx, py - cy)
    return t, cx, cy, d


def track_dtg_nm(
    cur_lat: float, cur_lon: float, points_latlon: List[Tuple[float, float]]
) -> Optional[float]:
    """
    Remaining NM along the polyline route from the current position projected onto the nearest segment.
    """
    if not points_latlon or len(points_latlon) < 2:
        return None

    ref_lat = points_latlon[0][0]
    ref_lon = points_latlon[0][1]

    xy = [ll_to_nm_xy(lat, lon, ref_lat, ref_lon) for lat, lon in points_latlon]

    seg_len: List[float] = []
    cum: List[float] = [0.0]
    total = 0.0
    for i in range(len(xy) - 1):
        ax, ay = xy[i]
        bx, by = xy[i + 1]
        L = math.hypot(bx - ax, by - ay)
        seg_len.append(L)
        total += L
        cum.append(total)

    px, py = ll_to_nm_xy(cur_lat, cur_lon, ref_lat, ref_lon)

    best_dist = None
    best_along = None

    for i in range(len(xy) - 1):
        ax, ay = xy[i]
        bx, by = xy[i + 1]
        t, _, _, d = project_point_to_segment_nm(px, py, ax, ay, bx, by)
        along = cum[i] + t * seg_len[i]

        if best_dist is None or d < best_dist:
            best_dist = d
            best_along = along

    if best_along is None:
        return None

    return max(0.0, total - best_along)


def hours_to_hhmm(hours: Optional[float]) -> Optional[str]:
    if hours is None:
        return None
    try:
        total_minutes = int(round(hours * 60))
        if total_minutes < 0:
            return None
        h = total_minutes // 60
        m = total_minutes % 60
        return f"{h:02d}:{m:02d}"
    except Exception:
        return None


def eta_zulu_from_hours(hours: Optional[float]) -> Optional[str]:
    if hours is None:
        return None
    try:
        now = datetime.now(timezone.utc)
        eta = now + timedelta(hours=float(hours))
        return eta.strftime("%H:%MZ")
    except Exception:
        return None


def build_route_points_from_simbrief(sb_raw: Optional[dict]) -> List[Tuple[float, float]]:
    """
    Build polyline: origin -> navlog points -> destination.
    Uses pos_lat/pos_long fields when present.
    """
    if not sb_raw:
        return []

    points: List[Tuple[float, float]] = []

    o_lat = to_float(deep_get(sb_raw, "origin", "pos_lat"))
    o_lon = to_float(deep_get(sb_raw, "origin", "pos_long"))
    if o_lat is not None and o_lon is not None:
        points.append((o_lat, o_lon))

    navlog = sb_raw.get("navlog")
    if isinstance(navlog, list):
        for wp in navlog:
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

    # de-dup consecutive identical points (can happen)
    cleaned: List[Tuple[float, float]] = []
    for p in points:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)

    return cleaned


def normalize_aircraft_icao(v: Any) -> Optional[str]:
    s = json_safe(v)
    if s is None:
        return None
    s = str(s).strip().upper()

    # reject known placeholder / garbage values
    if s in {"", "NONE", "NULL", "N/A", "TEXT", "STRING"}:
        return None

    m = re.search(r"\b([A-Z0-9]{3,4})\b", s)
    if not m:
        return None
    return normalize_icao(m.group(1), max_len=4)


def get_aircraft_icao_from_simconnect(aq_obj: Optional[AircraftRequests]) -> Optional[str]:
    # Try a few common ones (some aircraft expose different values)
    candidates = [
        "ATC_MODEL",
        "ATC MODEL",
        "ATC_TYPE",
        "ATC TYPE",
        "TITLE",  # last resort (we extract a 3-4 char token if possible)
    ]
    for name in candidates:
        v = safe_get(aq_obj, name, None)
        c = normalize_aircraft_icao(v)
        if c:
            return c
    return None


# -----------------------------
# Data model
# -----------------------------
def get_state() -> Dict[str, Any]:
    aq_obj, sim_ok, sim_msg = ensure_connection()

    # live sim vars (underscore names)
    ias = safe_get(aq_obj, "AIRSPEED_INDICATED", None)
    alt = safe_get(aq_obj, "PLANE_ALTITUDE", None)
    vs = safe_get(aq_obj, "VERTICAL_SPEED", None)
    gs = safe_get(aq_obj, "GROUND_VELOCITY", None)

    hdg_act = rad_to_deg(safe_get(aq_obj, "PLANE_HEADING_DEGREES_MAGNETIC")) or rad_to_deg(
        safe_get(aq_obj, "PLANE_HEADING_DEGREES_TRUE")
    )

    oat_c = safe_get(aq_obj, "AMBIENT_TEMPERATURE", None)
    tat_c = safe_get(aq_obj, "TOTAL_AIR_TEMPERATURE", None)

    wind_dir_true = safe_get(aq_obj, "AMBIENT_WIND_DIRECTION", None)
    wind_spd_kt = safe_get(aq_obj, "AMBIENT_WIND_VELOCITY", None)

    # aircraft ICAO (from simconnect if possible)
    ac_icao_live = normalize_aircraft_icao(safe_get(aq_obj, "ATC_MODEL", None))

    # current position for DTG calc
    cur_lat = to_float(safe_get(aq_obj, "PLANE_LATITUDE", None))
    cur_lon = to_float(safe_get(aq_obj, "PLANE_LONGITUDE", None))

    # simbrief
    sb_raw, sb_err = get_simbrief_cached()
    sb_fields = extract_simbrief_fields(sb_raw) if sb_raw else {}

    # airline theme resolution
    airline_icao, airline_source = resolve_airline_icao(sb_fields)

    # aircraft ICAO source resolution
    # aircraft ICAO: prefer SimBrief (reliable), fallback to SimConnect
    ac_icao_sb = normalize_icao(sb_fields.get("aircraft_icao"), max_len=4)

    # IMPORTANT: if SimConnect returns placeholder junk like "TEXT", treat as None
    if ac_icao_live in {"TEXT", "STRING", "NONE", "NULL", ""}:
        ac_icao_live = None

    aircraft_icao = ac_icao_sb or ac_icao_live
    aircraft_icao_source = "simbrief" if ac_icao_sb else ("simconnect" if ac_icao_live else "none")

    # DTG/ETE/ETA (track miles preferred, fallback to direct)
    dtg_nm = None
    if sb_raw and cur_lat is not None and cur_lon is not None:
        route_points = build_route_points_from_simbrief(sb_raw)
        if len(route_points) >= 2:
            dtg_nm = track_dtg_nm(cur_lat, cur_lon, route_points)

    # fallback: direct to destination if we have coords
    if dtg_nm is None and sb_raw and cur_lat is not None and cur_lon is not None:
        d_lat = to_float(deep_get(sb_raw, "destination", "pos_lat"))
        d_lon = to_float(deep_get(sb_raw, "destination", "pos_long"))
        if d_lat is not None and d_lon is not None:
            dtg_nm = haversine_nm(cur_lat, cur_lon, d_lat, d_lon)

    gs_kt = to_float(gs)
    ete_h = None
    if dtg_nm is not None and gs_kt is not None and gs_kt >= 1.0:
        ete_h = dtg_nm / gs_kt

    ete_hhmm = hours_to_hhmm(ete_h)
    eta_z = eta_zulu_from_hours(ete_h)

    state = {
        "simconnect_ok": sim_ok,
        "simconnect_msg": sim_msg,
        # live
        "ias_kt": ias,
        "gs_kt": gs,
        "alt_ft": alt,
        "vs_fpm": vs,
        "hdg_act": hdg_act,
        "oat_c": oat_c,
        "tat_c": tat_c,
        "wind_dir_true": wind_dir_true,
        "wind_spd_kt": wind_spd_kt,
        # flight progress
        "dtg_nm": dtg_nm,
        "ete_hhmm": ete_hhmm,
        "eta_z": eta_z,
        # ✅ BACKWARD COMPAT (overlay expects these)
        "ete": ete_hhmm,
        "eta": eta_z,
        # "ete_hhmm": ete_hhmm,
        # "eta_z": eta_z,
        # simbrief
        "simbrief_ok": sb_raw is not None,
        "simbrief_error": sb_err,
        "simbrief": sb_fields,
        # resolved airline theme key
        "airline_icao": airline_icao,
        "airline_source": airline_source,
        # aircraft type (ICAO)
        "aircraft_icao": aircraft_icao,
        "aircraft_icao_source": aircraft_icao_source,
    }

    return {k: json_safe(v) for k, v in state.items()}


# -----------------------------
# Routes
# -----------------------------
@app.get("/data")
def data():
    return jsonify(get_state())


@app.get("/simbrief")
def simbrief():
    sb_raw, sb_err = get_simbrief_cached()
    return jsonify(
        {
            "ok": sb_raw is not None,
            "error": sb_err,
            "data": extract_simbrief_fields(sb_raw) if sb_raw else {},
        }
    )


@app.get("/theme")
def theme():
    sb_raw, _ = get_simbrief_cached()
    sb_fields = extract_simbrief_fields(sb_raw) if sb_raw else {}
    airline_icao, source = resolve_airline_icao(sb_fields)

    theme_obj = load_theme_for_icao(airline_icao)

    logo_url = None
    if theme_obj.get("logo"):
        p = THEMES_DIR / theme_obj["icao"] / theme_obj["logo"]
        if p.exists():
            logo_url = f"/themes/{theme_obj['icao']}/{theme_obj['logo']}"

    return jsonify(
        {
            "airline_icao": theme_obj.get("icao", "default"),
            "source": source,
            "theme": theme_obj,
            "logo_url": logo_url,
        }
    )


@app.get("/themes/<icao>/<path:filename>")
def themes_static(icao: str, filename: str):
    safe_icao = "".join(ch for ch in icao.upper() if ch.isalnum())[:3] or "default"
    directory = THEMES_DIR / safe_icao
    return send_from_directory(directory, filename)


@app.get("/overlay")
def overlay():
    return send_file("overlay.html")


@app.get("/simbrief_raw_airports")
def simbrief_raw_airports():
    sb_raw, sb_err = get_simbrief_cached()
    if not sb_raw:
        return jsonify({"ok": False, "error": sb_err})
    return jsonify(
        {
            "origin": sb_raw.get("origin"),
            "destination": sb_raw.get("destination"),
        }
    )


@app.get("/simbrief_raw")
def simbrief_raw():
    sb_raw, sb_err = get_simbrief_cached()
    if not sb_raw:
        return jsonify({"ok": False, "error": sb_err})

    # keep it small + useful (no giant NOTAM dumps)
    return jsonify(
        {
            "ok": True,
            "error": None,
            "general": sb_raw.get("general"),
            "atc": sb_raw.get("atc"),
            "origin": {
                "icao_code": deep_get(sb_raw, "origin", "icao_code"),
                "metar": deep_get(sb_raw, "origin", "metar"),
                "taf": deep_get(sb_raw, "origin", "taf"),
                "plan_rwy": deep_get(sb_raw, "origin", "plan_rwy"),
                "name": deep_get(sb_raw, "origin", "name"),
            },
            "destination": {
                "icao_code": deep_get(sb_raw, "destination", "icao_code"),
                "metar": deep_get(sb_raw, "destination", "metar"),
                "taf": deep_get(sb_raw, "destination", "taf"),
                "plan_rwy": deep_get(sb_raw, "destination", "plan_rwy"),
                "name": deep_get(sb_raw, "destination", "name"),
            },
        }
    )


@app.get("/")
def root():
    return send_file("overlay.html")


if __name__ == "__main__":
    THEMES_DIR.mkdir(parents=True, exist_ok=True)
    (THEMES_DIR / "default").mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
