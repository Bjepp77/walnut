"""Flask API + static UI for the Walnut dashboard."""

import logging
import subprocess
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import permissions
import stt
import store
import voices as voice_lib

HERE = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(HERE / "static"))
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# wired up by app.py
CORE = None
HOTKEYS = None

# Walnut binds 127.0.0.1, but a page in your browser can still POST to it, and
# DNS rebinding can make an attacker's domain resolve there. Both are defeated
# by refusing any request whose Host isn't loopback.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


@app.before_request
def _guard_host():
    host = (request.host or "").rsplit(":", 1)[0]
    if host not in _ALLOWED_HOSTS:
        return jsonify({"error": "forbidden host"}), 403
    origin = request.headers.get("Origin")
    if origin and origin not in _allowed_origins(request.environ.get("SERVER_PORT", "")):
        return jsonify({"error": "forbidden origin"}), 403


def _allowed_origins(port: str) -> set[str]:
    """The dashboard's own origins, exact: scheme, loopback host and the port
    this server listens on. A prefix test let http://localhost.evil.example in."""
    hosts = ("127.0.0.1", "localhost", "[::1]")
    allowed = {f"http://{h}:{port}" for h in hosts}
    if port == "80":  # browsers omit the default port from Origin
        allowed |= {f"http://{h}" for h in hosts}
    return allowed


@app.errorhandler(ValueError)
def _bad_input(e):
    """A rejected value is the caller's fault, not a server fault."""
    return jsonify({"error": str(e)}), 400


@app.errorhandler(Exception)
def _unhandled(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    app.logger.exception("unhandled error")
    return jsonify({"error": "internal error"}), 500


def _int_arg(name: str, default: int) -> int:
    raw = request.args.get(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {raw!r}")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------- dashboard

@app.route("/api/dashboard")
def dashboard():
    period = request.args.get("period", "week")
    if period not in ("week", "month", "all"):
        period = "week"
    return jsonify(store.stats(period))


# ---------------------------------------------------------------- dictionary

@app.route("/api/words", methods=["GET", "POST"])
def words():
    if request.method == "POST":
        w = (request.json or {}).get("word", "").strip()
        if w:
            store.words_add(w)
    return jsonify(store.words_list())


@app.route("/api/words/<int:wid>", methods=["DELETE"])
def words_delete(wid):
    store.words_delete(wid)
    return jsonify({"ok": True})


@app.route("/api/replacements", methods=["GET", "POST"])
def replacements():
    if request.method == "POST":
        d = request.json or {}
        if d.get("wrong", "").strip() and d.get("right", "").strip():
            store.replacements_add(d["wrong"], d["right"])
    return jsonify(store.replacements_list())


@app.route("/api/replacements/<int:rid>", methods=["DELETE"])
def replacements_delete(rid):
    store.replacements_delete(rid)
    return jsonify({"ok": True})


# ---------------------------------------------------------------- snippets

@app.route("/api/snippets", methods=["GET", "POST"])
def snippets():
    if request.method == "POST":
        d = request.json or {}
        if d.get("trigger", "").strip() and d.get("expansion", "").strip():
            store.snippets_add(d["trigger"], d["expansion"])
    return jsonify({
        "enabled": store.get("snippets_enabled") == "1",
        "snippets": store.snippets_list(),
    })


@app.route("/api/snippets/<int:sid>", methods=["PUT", "DELETE"])
def snippet_item(sid):
    if request.method == "DELETE":
        store.snippets_delete(sid)
    else:
        d = request.json or {}
        trigger = str(d.get("trigger", "")).strip()
        expansion = str(d.get("expansion", ""))
        if not trigger or not expansion:
            raise ValueError("trigger and expansion are required")
        store.snippets_update(sid, trigger, expansion,
                              1 if d.get("enabled", 1) else 0)
    return jsonify({"ok": True})


@app.route("/api/snippets/toggle", methods=["POST"])
def snippets_toggle():
    now = "0" if store.get("snippets_enabled") == "1" else "1"
    store.set_setting("snippets_enabled", now)
    return jsonify({"enabled": now == "1"})


# ---------------------------------------------------------------- history

@app.route("/api/history")
def history():
    return jsonify(store.history_list(
        q=request.args.get("q", ""),
        kind=request.args.get("type", ""),
        limit=_int_arg("limit", 200),   # `?limit=abc` used to be a 500
    ))


@app.route("/api/history/<int:hid>", methods=["DELETE"])
def history_delete(hid):
    store.history_delete(hid)
    return jsonify({"ok": True})


@app.route("/api/history/<int:hid>/replay", methods=["POST"])
def history_replay(hid):
    entry = store.history_get(hid)
    if not entry:
        return jsonify({"error": "not found"}), 404
    if not CORE:
        return jsonify({"error": "core not ready"}), 503
    audio = entry.get("audio_path")
    if audio and Path(audio).exists():
        # via CORE so the handle is kept and /api/speech/stop can cut it
        CORE.play_file(audio)
    else:
        threading.Thread(target=CORE.speak, args=(entry["text"], False),
                         daemon=True).start()
    return jsonify({"ok": True})


# ------------------------------------------------------------------- speech
# Only one thing can be audible at a time — speak() and play_file() each stop
# whatever preceded them — so the UI tracks a single "now playing" card.

@app.route("/api/speech/stop", methods=["POST"])
def speech_stop():
    if CORE:
        CORE.stop_all()
    return jsonify({"ok": True, "playing": False})


@app.route("/api/speech/status")
def speech_status():
    return jsonify({"playing": bool(CORE and CORE.busy_audio())})


@app.route("/api/history/<int:hid>/reveal", methods=["POST"])
def history_reveal(hid):
    entry = store.history_get(hid)
    audio = entry.get("audio_path") if entry else None
    if audio and Path(audio).exists():
        subprocess.run(["open", "-R", audio])
        return jsonify({"ok": True})
    return jsonify({"error": "no recording for this entry"}), 404


# ---------------------------------------------------------------- settings

@app.route("/api/settings", methods=["GET", "PUT"])
def settings():
    if request.method == "PUT":
        d = request.json or {}

        # Validate the whole payload before writing any of it. Previously a bad
        # value was persisted and only then interpreted: an unparseable hotkey
        # left the listener dead and crash-looped Walnut on the next launch.
        clean, hotkeys_changed = {}, False
        for key in store.DEFAULTS:
            if key in d:
                clean[key] = store.validate_setting(key, d[key])  # raises -> 400
                if key.startswith("hotkey_") and clean[key] != store.get(key):
                    hotkeys_changed = True

        previous = {k: store.get(k) for k in clean}
        for key, value in clean.items():
            store.set_setting(key, value)

        if hotkeys_changed and HOTKEYS:
            try:
                HOTKEYS.reload()
            except Exception as e:
                for key, value in previous.items():   # put it all back
                    store.set_setting(key, value)
                HOTKEYS.reload()
                raise ValueError(f"could not bind those hotkeys: {e}")

        # A lowered cap should bite now, not at the next dictation.
        if "recordings_keep" in clean:
            store.prune_recordings()
    return jsonify(store.all_settings())


@app.route("/api/system")
def system():
    """What hardware Walnut found, and which engine it chose."""
    info = stt.describe(store.get("stt_backend"))
    info["models"] = stt.catalog(info["backend"])
    return jsonify(info)


@app.route("/api/status")
def status():
    """Everything that could be silently wrong right now."""
    perms = permissions.summary()
    voice = voice_lib.status(store.get("tts_voice"), store.get("stt_language"))
    return jsonify({
        "accessibility": perms["accessibility"],
        "accessibility_hint": perms["hint"],
        "model_state": CORE.model_state if CORE else "unknown",
        "model_error": CORE.model_error if CORE else None,
        "model": stt.canonical(store.get("stt_model")),
        # Not an error — nobody knows macOS hides good voices behind a
        # download, so they conclude local narration just sounds like this.
        "voice_ok": voice["ok"],
        "voice_reason": voice["reason"],
        "voice_title": voice["title"],
        "voice_hint": voice["hint"],
        "voice_suggestions": voice["suggestions"],
        # Apple renamed this pane at macOS 26; the UI must not hardcode it.
        "voice_pane": voice_lib.pane_name(),
    })


@app.route("/api/permissions/open", methods=["POST"])
def permissions_open():
    permissions.open_accessibility_settings()
    return jsonify({"ok": True})


@app.route("/api/voices")
def voices():
    return jsonify(voice_lib.list_voices())


@app.route("/api/voices/open", methods=["POST"])
def voices_open():
    voice_lib.open_voice_settings()
    return jsonify({"ok": True})


@app.route("/api/test-voice", methods=["POST"])
def test_voice():
    if CORE:
        d = request.json or {}
        threading.Thread(
            target=CORE.speak,
            args=(d.get("text", "Walnut here. This is how I sound."), False),
            daemon=True).start()
    return jsonify({"ok": True})


def port_in_use(port: int) -> bool:
    """True if something already listens on 127.0.0.1:port."""
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run(port: int) -> None:
    # werkzeug calls sys.exit() when the port is taken. In a daemon thread that
    # kills the thread and nothing else, so the dashboard would vanish without
    # a word. Surface it instead.
    try:
        app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
    except SystemExit:
        raise RuntimeError(f"port {port} is already in use")
