import time
from typing import Any, Optional, Tuple

from SimConnect import AircraftRequests, SimConnect

from .nav import rad_to_deg, to_float
from .simbrief import normalize_aircraft_icao

sm = None
aq = None
_last_state: Optional[bool] = None
_last_msg_ts = 0.0


def _log_once(ok: bool, msg: str):
    global _last_state, _last_msg_ts
    now = time.time()
    if _last_state != ok or (now - _last_msg_ts) > 15:
        print(msg)
        _last_state = ok
        _last_msg_ts = now


def ensure_connection() -> Tuple[Optional[AircraftRequests], bool, Optional[str]]:
    global sm, aq
    try:
        if sm is None or aq is None:
            sm = SimConnect()
            aq = AircraftRequests(sm, _time=1000)
        _log_once(True, "[SimConnect] Connected.")
        return aq, True, None
    except Exception as e:
        sm = None
        aq = None
        msg = f"[SimConnect] Not available (is MSFS running?). {e}"
        _log_once(False, msg)
        return None, False, msg


def safe_get(aq_obj: Optional[AircraftRequests], name: str, default: Any = None) -> Any:
    if aq_obj is None:
        return default
    try:
        v = aq_obj.get(name)
        return default if v is None else v
    except Exception:
        return default


def get_aircraft_icao(aq_obj: Optional[AircraftRequests]) -> Optional[str]:
    for name in ["ATC_MODEL", "ATC MODEL", "ATC_TYPE", "ATC TYPE", "TITLE"]:
        c = normalize_aircraft_icao(safe_get(aq_obj, name))
        if c:
            return c
    return None
