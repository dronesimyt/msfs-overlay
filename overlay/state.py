from typing import Any, Dict

from .config import CONFIG, get_cfg
from .nav import eta_zulu_from_hours, haversine_nm, hours_to_hhmm, rad_to_deg, to_float, track_dtg_nm
from .simbrief import (
    build_route_points_from_simbrief,
    deep_get,
    extract_simbrief_fields,
    get_simbrief_cached,
    json_safe,
    normalize_icao,
    resolve_airline_icao,
)
from .simconnect import ensure_connection, get_aircraft_icao, safe_get
from .tokcount import TIKTOK_USER_ID, TOKCOUNT_REFRESH_SECONDS, get_tokcount


def get_state() -> Dict[str, Any]:
    aq_obj, sim_ok, sim_msg = ensure_connection()

    ias = safe_get(aq_obj, "AIRSPEED_INDICATED")
    alt = safe_get(aq_obj, "PLANE_ALTITUDE")
    vs = safe_get(aq_obj, "VERTICAL_SPEED")
    gs = safe_get(aq_obj, "GROUND_VELOCITY")
    hdg_act = rad_to_deg(safe_get(aq_obj, "PLANE_HEADING_DEGREES_MAGNETIC")) or rad_to_deg(
        safe_get(aq_obj, "PLANE_HEADING_DEGREES_TRUE")
    )
    oat_c = safe_get(aq_obj, "AMBIENT_TEMPERATURE")
    tat_c = safe_get(aq_obj, "TOTAL_AIR_TEMPERATURE")
    wind_dir_true = safe_get(aq_obj, "AMBIENT_WIND_DIRECTION")
    wind_spd_kt = safe_get(aq_obj, "AMBIENT_WIND_VELOCITY")
    cur_lat = to_float(safe_get(aq_obj, "PLANE_LATITUDE"))
    cur_lon = to_float(safe_get(aq_obj, "PLANE_LONGITUDE"))

    ac_icao_live = get_aircraft_icao(aq_obj)
    if ac_icao_live in {"TEXT", "STRING", "NONE", "NULL", ""}:
        ac_icao_live = None

    sb_raw, sb_err = get_simbrief_cached()
    sb_fields = extract_simbrief_fields(sb_raw) if sb_raw else {}
    airline_icao, airline_source = resolve_airline_icao(sb_fields)

    ac_icao_sb = normalize_icao(sb_fields.get("aircraft_icao"), max_len=4)
    aircraft_icao = ac_icao_sb or ac_icao_live
    aircraft_icao_source = "simbrief" if ac_icao_sb else ("simconnect" if ac_icao_live else "none")

    dtg_nm = None
    if sb_raw and cur_lat is not None and cur_lon is not None:
        route_points = build_route_points_from_simbrief(sb_raw)
        if len(route_points) >= 2:
            dtg_nm = track_dtg_nm(cur_lat, cur_lon, route_points)

    if dtg_nm is None and sb_raw and cur_lat is not None and cur_lon is not None:
        d_lat = to_float(deep_get(sb_raw, "destination", "pos_lat"))
        d_lon = to_float(deep_get(sb_raw, "destination", "pos_long"))
        if d_lat is not None and d_lon is not None:
            dtg_nm = haversine_nm(cur_lat, cur_lon, d_lat, d_lon)

    gs_kt = to_float(gs)
    ete_h = (dtg_nm / gs_kt) if (dtg_nm is not None and gs_kt is not None and gs_kt >= 1.0) else None
    ete_hhmm = hours_to_hhmm(ete_h)
    eta_z = eta_zulu_from_hours(ete_h)

    tok = get_tokcount()

    state = {
        "simconnect_ok": sim_ok,
        "simconnect_msg": sim_msg,
        "ias_kt": ias,
        "gs_kt": gs,
        "alt_ft": alt,
        "vs_fpm": vs,
        "hdg_act": hdg_act,
        "oat_c": oat_c,
        "tat_c": tat_c,
        "wind_dir_true": wind_dir_true,
        "wind_spd_kt": wind_spd_kt,
        "dtg_nm": dtg_nm,
        "ete_hhmm": ete_hhmm,
        "eta_z": eta_z,
        "ete": ete_hhmm,
        "eta": eta_z,
        "simbrief_ok": sb_raw is not None,
        "simbrief_error": sb_err,
        "simbrief": sb_fields,
        "airline_icao": airline_icao,
        "airline_source": airline_source,
        "aircraft_icao": aircraft_icao,
        "aircraft_icao_source": aircraft_icao_source,
        "tiktok_user_id": CONFIG.get("tiktok_user_id"),
        "tiktok_followers_goal": int(get_cfg("tiktok_followers_goal", 1000)),
        "tokcount_refresh_seconds": TOKCOUNT_REFRESH_SECONDS,
        "tokcount_error": tok.get("error"),
        "tokcount_raw": {
            "followers": tok.get("followers"),
            "likes": tok.get("likes"),
            "following": tok.get("following"),
            "videos": tok.get("videos"),
            "ts": tok.get("ts"),
            "error": tok.get("error"),
        },
    }

    return {k: json_safe(v) for k, v in state.items()}
