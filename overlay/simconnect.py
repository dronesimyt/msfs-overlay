import threading
import time
from typing import Any, Optional, Tuple

from SimConnect import AircraftRequests, SimConnect

from .nav import rad_to_deg, to_float
from .simbrief import normalize_aircraft_icao

# Retry interval while MSFS is not reachable, and how long a connection may
# deliver no data before it is considered dead (e.g. after an MSFS crash).
RECONNECT_INTERVAL_SECONDS = 5.0
STALE_DATA_SECONDS = 15.0

sm = None
aq = None
_lock = threading.Lock()
_last_attempt_ts = 0.0
_last_data_ts = 0.0
_last_error: Optional[str] = None
_last_state: Optional[bool] = None
_last_msg_ts = 0.0


def _log_once(ok: bool, msg: str):
    global _last_state, _last_msg_ts
    now = time.time()
    if _last_state != ok or (now - _last_msg_ts) > 15:
        print(msg)
        _last_state = ok
        _last_msg_ts = now


def _disconnect():
    global sm, aq
    old = sm
    sm = None
    aq = None
    if old is None:
        return
    try:
        old.quit = 1
        thread = getattr(old, "timerThread", None)
        if thread is not None:
            thread.join(timeout=1.0)
        old.dll.Close(old.hSimConnect)
    except Exception:
        pass


def report_data(received: bool):
    """Called after each poll; drops the connection if MSFS stopped delivering data."""
    global _last_data_ts
    now = time.time()
    with _lock:
        if sm is None:
            return
        if received:
            _last_data_ts = now
        elif (now - _last_data_ts) > STALE_DATA_SECONDS:
            _log_once(False, "[SimConnect] No data received, reconnecting.")
            _disconnect()


def ensure_connection() -> Tuple[Optional[AircraftRequests], bool, Optional[str]]:
    global sm, aq, _last_attempt_ts, _last_data_ts, _last_error
    with _lock:
        # MSFS signals a regular shutdown via SIMCONNECT_RECV_ID_QUIT.
        if sm is not None and getattr(sm, "quit", 0):
            _disconnect()
            _last_error = "[SimConnect] MSFS closed the connection."
            _log_once(False, _last_error)

        if sm is not None and aq is not None:
            return aq, True, None

        now = time.time()
        if (now - _last_attempt_ts) < RECONNECT_INTERVAL_SECONDS:
            return None, False, _last_error or "[SimConnect] Not connected."
        _last_attempt_ts = now

        try:
            sm = SimConnect()
            aq = AircraftRequests(sm, _time=1000)
            _last_data_ts = now
            _last_error = None
            _log_once(True, "[SimConnect] Connected.")
            return aq, True, None
        except Exception as e:
            _disconnect()
            _last_error = f"[SimConnect] Not available (is MSFS running?). {e}"
            _log_once(False, _last_error)
            return None, False, _last_error


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
