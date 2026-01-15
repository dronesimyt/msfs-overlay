import math
import os
import time
import threading
import json
from pathlib import Path

import requests
from flask import Flask, jsonify, send_file
from SimConnect import SimConnect, AircraftRequests

app = Flask(__name__)

sm = None
aq = None

# ---------------- config.json (preferred over env vars) ----------------
CONFIG_PATH = Path(__file__).with_name("config.json")

def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

CONFIG = load_config()

def get_simbrief_identity():
    # priority: config.json > env vars
    username = CONFIG.get("simbrief_username") or os.getenv("SIMBRIEF_USERNAME")
    userid = CONFIG.get("simbrief_userid") or os.getenv("SIMBRIEF_USERID")
    return username, userid

# ---------------- SimBrief cache ----------------
SIMBRIEF_CACHE_SECONDS = 60
_simbrief_cache = {"ts": 0, "data": None, "error": None}
_simbrief_lock = threading.Lock()

# ---------------- SimConnect helpers ----------------
def ensure_connection():
    global sm, aq
    if sm is None:
        sm = SimConnect()
        aq = AircraftRequests(sm, _time=200)  # ~5Hz cache
    return aq

def safe_get(aq_obj, name, default=None):
    try:
        v = aq_obj.get(name)
        return default if v is None else v
    except Exception:
        return default

def rad_to_deg(v):
    if v is None:
        return None
    try:
        return math.degrees(float(v)) % 360.0
    except (TypeError, ValueError):
        return None

def meters_to_nm(v):
    if v is None:
        return None
    try:
        return float(v) / 1852.0
    except (TypeError, ValueError):
        return None

def seconds_to_hhmm(seconds):
    if seconds is None:
        return None
    try:
        s = int(float(seconds))
        if s < 0:
            return None
        h = s // 3600
        m = (s % 3600) // 60
        return f"{h:02d}:{m:02d}"
    except (TypeError, ValueError):
        return None

def json_safe(v):
    # bytes -> string (fixes your 500 error)
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore").strip("\x00").strip()
        except Exception:
            return str(v)

    if v is None or isinstance(v, (str, int, float, bool)):
        return v

    return str(v)

# ---------------- SimBrief fetch + parse ----------------
def _fetch_simbrief_latest():
    base = "https://www.simbrief.com/api/xml.fetcher.php"
    params = {"json": "1"}

    username, userid = get_simbrief_identity()

    if userid:
        params["userid"] = userid
    elif username:
        params["username"] = username
    else:
        return None, "SIMBRIEF username/userid not configured (config.json or env var)"

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

def extract_simbrief_fields(sb):
    if not sb:
        return {}

    def deep_get(d, *path):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    out = {}

    # origin/destination
    out["dep_icao"] = deep_get(sb, "origin", "icao_code") or deep_get(sb, "origin", "icao")
    out["arr_icao"] = deep_get(sb, "destination", "icao_code") or deep_get(sb, "destination", "icao")

    # callsign/flight
    out["callsign"] = deep_get(sb, "atc", "callsign")

    # aircraft
    out["aircraft"] = deep_get(sb, "aircraft", "icao_code") or deep_get(sb, "aircraft", "name")

    # cruise
    out["cruise"] = deep_get(sb, "general", "initial_altitude") or deep_get(sb, "general", "cruise_altitude")

    # planned dist / ete (these vary a lot, so keep it best-effort)
    out["planned_distance"] = deep_get(sb, "general", "distance") or deep_get(sb, "general", "route_distance")
    out["planned_ete"] = deep_get(sb, "times", "enroute_time") or deep_get(sb, "general", "ete")

    return {k: json_safe(v) for k, v in out.items() if v is not None}

# ---------------- YOUR get_state() (expanded + simbrief merged) ----------------
def get_state():
    aq_obj = ensure_connection()

    # Your working basics
    ias = safe_get(aq_obj, "AIRSPEED_INDICATED", 0.0)
    alt = safe_get(aq_obj, "PLANE_ALTITUDE", 0.0)
    vs = safe_get(aq_obj, "VERTICAL_SPEED", 0.0)

    # Add GS
    gs = safe_get(aq_obj, "GROUND_VELOCITY", 0.0)

    # ACTUAL heading
    hdg_act = (
        rad_to_deg(safe_get(aq_obj, "PLANE_HEADING_DEGREES_MAGNETIC"))
        or rad_to_deg(safe_get(aq_obj, "PLANE_HEADING_DEGREES_TRUE"))
    )

    # Temperatures (OAT/TAT)
    oat_c = safe_get(aq_obj, "AMBIENT_TEMPERATURE")          # Celsius
    tat_c = safe_get(aq_obj, "TOTAL_AIR_TEMPERATURE")        # Celsius

    # Wind
    wind_dir_true = safe_get(aq_obj, "AMBIENT_WIND_DIRECTION")  # degrees true
    wind_spd_kt = safe_get(aq_obj, "AMBIENT_WIND_VELOCITY")     # knots

    # GPS next waypoint
    # wp_next = json_safe(safe_get(aq_obj, "GPS_WP_NEXT_ID"))
    wp_dist_nm = meters_to_nm(safe_get(aq_obj, "GPS_WP_DISTANCE"))
    wp_ete = seconds_to_hhmm(safe_get(aq_obj, "GPS_WP_ETE"))
    wp_eta_sec = safe_get(aq_obj, "GPS_WP_ETA")

    # SimBrief (cached)
    sb_raw, sb_err = get_simbrief_cached()
    sb_fields = extract_simbrief_fields(sb_raw) if sb_raw else {}

    state = {
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
        # "wp_next": wp_next,
        "wp_dist_nm": wp_dist_nm,
        "wp_ete": wp_ete,
        "wp_eta_sec": wp_eta_sec,

        # simbrief
        "simbrief_ok": sb_raw is not None,
        "simbrief_error": sb_err,
        "simbrief": sb_fields,
    }

    return {k: json_safe(v) for k, v in state.items()}

# ---------------- routes ----------------
@app.get("/data")
def data():
    return jsonify(get_state())

@app.get("/simbrief")
def simbrief():
    sb_raw, sb_err = get_simbrief_cached()
    return jsonify({
        "ok": sb_raw is not None,
        "error": sb_err,
        "data": extract_simbrief_fields(sb_raw) if sb_raw else {},
    })

@app.get("/overlay")
def overlay():
    return send_file("overlay.html")

@app.get("/")
def root():
    return send_file("overlay.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
