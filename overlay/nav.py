import math
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple


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


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_nm = 3440.065
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r_nm * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def ll_to_nm_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    return (lon - ref_lon) * math.cos(math.radians(ref_lat)) * 60.0, (lat - ref_lat) * 60.0


def project_point_to_segment_nm(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float, float]:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return 0.0, ax, ay, math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    cx, cy = ax + t * vx, ay + t * vy
    return t, cx, cy, math.hypot(px - cx, py - cy)


def track_dtg_nm(
    cur_lat: float, cur_lon: float, points_latlon: List[Tuple[float, float]]
) -> Optional[float]:
    if not points_latlon or len(points_latlon) < 2:
        return None

    ref_lat, ref_lon = points_latlon[0]
    xy = [ll_to_nm_xy(lat, lon, ref_lat, ref_lon) for lat, lon in points_latlon]

    seg_len: List[float] = []
    cum: List[float] = [0.0]
    total = 0.0
    for i in range(len(xy) - 1):
        L = math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
        seg_len.append(L)
        total += L
        cum.append(total)

    px, py = ll_to_nm_xy(cur_lat, cur_lon, ref_lat, ref_lon)
    best_dist = None
    best_along = None

    for i in range(len(xy) - 1):
        t, _, _, d = project_point_to_segment_nm(px, py, *xy[i], *xy[i + 1])
        along = cum[i] + t * seg_len[i]
        if best_dist is None or d < best_dist:
            best_dist = d
            best_along = along

    return max(0.0, total - best_along) if best_along is not None else None


def hours_to_hhmm(hours: Optional[float]) -> Optional[str]:
    if hours is None:
        return None
    try:
        total_minutes = int(round(hours * 60))
        if total_minutes < 0:
            return None
        return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
    except Exception:
        return None


def eta_zulu_from_hours(hours: Optional[float]) -> Optional[str]:
    if hours is None:
        return None
    try:
        eta = datetime.now(timezone.utc) + timedelta(hours=float(hours))
        return eta.strftime("%H:%MZ")
    except Exception:
        return None
