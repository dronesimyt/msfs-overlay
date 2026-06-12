import logging
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from overlay.state import get_state
from overlay.themes import THEMES_DIR, load_theme_for_icao

app = Flask(__name__)


@app.get("/")
def overlay():
    return send_file("overlay.html")


@app.get("/data")
def data():
    return jsonify(get_state())


@app.post("/shutdown")
def shutdown():
    if request.remote_addr != "127.0.0.1":
        return jsonify({"error": "forbidden"}), 403
    threading.Timer(0.1, lambda: os._exit(0)).start()
    return jsonify({"ok": True})


@app.get("/debug")
def debug():
    if request.remote_addr != "127.0.0.1":
        return jsonify({"error": "forbidden"}), 403
    from overlay.tokcount import TIKTOK_USER_ID, TOKCOUNT_REFRESH_SECONDS, get_tokcount
    from overlay.simconnect import ensure_connection
    tok = get_tokcount()
    _, sim_ok, sim_msg = ensure_connection()
    return jsonify({
        "simconnect_ok": sim_ok,
        "simconnect_msg": sim_msg,
        "tiktok_user_id": TIKTOK_USER_ID,
        "tokcount_refresh_seconds": TOKCOUNT_REFRESH_SECONDS,
        "tokcount_error": tok.get("error"),
        "tokcount_raw": tok,
    })


@app.get("/theme")
def theme():
    from overlay.simbrief import extract_simbrief_fields, get_simbrief_cached, resolve_airline_icao
    sb_raw, _ = get_simbrief_cached()
    sb_fields = extract_simbrief_fields(sb_raw) if sb_raw else {}
    airline_icao, source = resolve_airline_icao(sb_fields)
    theme_obj = load_theme_for_icao(airline_icao)

    logo_url = None
    logo_name = theme_obj.get("logo")
    theme_dir = THEMES_DIR / theme_obj.get("icao", "default")

    if logo_name and (theme_dir / logo_name).exists():
        logo_url = f"/themes/{theme_obj['icao']}/{logo_name}"

    if not logo_url and theme_dir.exists():
        for candidate in ["logo.png", "logo_white.png", "logo_black.png"]:
            if (theme_dir / candidate).exists():
                logo_url = f"/themes/{theme_obj['icao']}/{candidate}"
                break

    if not logo_url and theme_dir.exists():
        for p in theme_dir.glob("logo*.*"):
            if p.is_file():
                logo_url = f"/themes/{theme_obj['icao']}/{p.name}"
                break

    return jsonify({
        "airline_icao": theme_obj.get("icao", "default"),
        "source": source,
        "theme": theme_obj,
        "logo_url": logo_url,
    })


@app.get("/themes/<icao>/<path:filename>")
def themes_static(icao: str, filename: str):
    safe_icao = "".join(ch for ch in icao.upper() if ch.isalnum())[:3] or "default"
    return send_from_directory(THEMES_DIR / safe_icao, filename)


if __name__ == "__main__":
    THEMES_DIR.mkdir(parents=True, exist_ok=True)
    (THEMES_DIR / "default").mkdir(parents=True, exist_ok=True)

    log_path = Path(__file__).with_name("overlay.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    app.run(host="127.0.0.1", port=5000, debug=False)
