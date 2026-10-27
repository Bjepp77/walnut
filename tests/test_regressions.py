"""Regression tests. Every test here is a bug Walnut actually shipped.

Run with:  uv run --group dev pytest -q
"""

import os
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _open_fds() -> int:
    out = subprocess.run(["lsof", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout
    return out.count("\n")


# ------------------------------------------------------------------- storage

def test_connections_are_closed(db):
    """`with _conn() as c` used sqlite3's context manager, which commits but
    never closes. Hundreds of dashboard polls leaked file descriptors."""
    before = _open_fds()
    for _ in range(300):
        db.get("stt_model")
    assert _open_fds() - before < 5


def test_wal_and_indexes(db):
    con = sqlite3.connect(db.DB_PATH)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    idx = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='history'")}
    assert "idx_history_ts" in idx
    con.close()


def test_history_search_escapes_like_wildcards(db):
    db.history_add("dictation", "literally 100% done", 1.0)
    db.history_add("dictation", "unrelated", 1.0)
    assert len(db.history_list(q="100%")) == 1
    assert len(db.history_list(q="%")) == 1        # not a match-everything


def test_history_limit_is_clamped(db):
    for i in range(5):
        db.history_add("dictation", f"row {i}", 1.0)
    assert len(db.history_list(limit=10**9)) == 5  # no unbounded fetch
    assert len(db.history_list(limit=-1)) == 1


# ---------------------------------------------------------------- vocabulary

def test_backslash_in_fixup_does_not_break_dictation(db):
    """A fix-up containing \\1 or \\t was passed to re.sub as a template.
    It raised re.PatternError inside a worker thread, so *every* subsequent
    transcription silently produced nothing."""
    from core import Vocabulary

    db.replacements_add("num", r"\1")
    db.replacements_add("path", r"C:\temp")
    out, _ = Vocabulary.apply("num and path here")
    assert out == r"\1 and C:\temp here"


def test_snippet_still_expands(db):
    from core import Vocabulary

    db.snippets_add("insert my sig", "me@example.com")
    # matching ignores case and trailing punctuation
    out, used = Vocabulary.apply("Insert my sig.")
    assert (out, used) == ("me@example.com", True)


def test_duplicate_snippet_trigger_first_one_wins(db):
    """Nothing stops two snippets sharing a trigger; the lower id wins because
    snippets_list() orders by trigger then returns on the first match. Pinned
    here so the behaviour is a decision rather than an accident."""
    from core import Vocabulary

    db.snippets_add("dup trigger", "first")
    db.snippets_add("dup trigger", "second")
    out, used = Vocabulary.apply("dup trigger")
    assert (out, used) == ("first", True)


# ------------------------------------------------------------------ settings

@pytest.mark.parametrize("key,value,ok", [
    ("tts_rate", "210", True),
    ("tts_rate", "fast", False),
    ("tts_rate", "9999", False),
    ("typing_wpm", "0", False),
    ("typing_wpm", "60", True),
    ("stt_backend", "bogus", False),
    ("stt_backend", "auto", True),
    ("stt_language", "english", False),
    ("stt_language", "en", True),
    ("hotkey_speak", "garbage", False),
    ("hotkey_speak", "<ctrl>+<alt>+s", True),
])
def test_validate_setting(db, key, value, ok):
    if ok:
        db.validate_setting(key, value)
    else:
        with pytest.raises(ValueError):
            db.validate_setting(key, value)


def test_unknown_setting_rejected(db):
    with pytest.raises(ValueError):
        db.validate_setting("rm_rf", "1")


def test_get_valid_heals_a_poisoned_database(db):
    """A database hand-edited (or written by an older Walnut) must not be able
    to take the process down on the next launch."""
    db.set_setting("hotkey_speak", "garbage")
    db.set_setting("tts_rate", "fast")
    assert db.get_valid("hotkey_speak") == db.DEFAULTS["hotkey_speak"]
    assert db.get_valid("tts_rate") == db.DEFAULTS["tts_rate"]


def test_hotkey_manager_survives_unparseable_combo(db):
    """hotkeys.start() ran in WalnutApp.__init__. A bad stored combo raised,
    the process died, and launchd's KeepAlive respawned it forever."""
    import core

    db.set_setting("hotkey_speak", "garbage")
    hk = core.HotkeyManager(core.Core())
    hk.start()                       # must not raise
    assert hk.listener is not None   # and must leave working hotkeys
    hk.stop()


# --------------------------------------------------------------------- audio

def test_audio_filenames_are_unique(db):
    """Second-resolution names collided: two dictations in one second wrote the
    same file, and deleting either unlinked the survivor's recording."""
    import core

    paths = {core.Core._save_audio(np.zeros(1600, dtype=np.float32))
             for _ in range(3)}
    assert len(paths) == 3


# ----------------------------------------------------------------- retention

def _wav(db, name: str):
    p = db.RECORDINGS / name
    p.write_bytes(b"RIFF")          # contents irrelevant; only the file matters
    return str(p)


def test_prune_keeps_only_the_newest_n(db):
    for i in range(6):
        db.history_add("dictation", f"clip {i}", 1.0,
                       audio_path=_wav(db, f"d{i}.wav"))
    assert db.prune_recordings(keep=3) == 3

    on_disk = sorted(p.name for p in db.RECORDINGS.glob("*.wav"))
    assert on_disk == ["d3.wav", "d4.wav", "d5.wav"]     # newest three survive

    rows = db.history_list()
    assert len(rows) == 6                                 # transcripts all kept
    with_audio = [r for r in rows if r["audio_path"]]
    assert len(with_audio) == 3


def test_prune_zero_keeps_transcripts_but_no_audio(db):
    db.history_add("dictation", "only text", 1.0, audio_path=_wav(db, "a.wav"))
    assert db.prune_recordings(keep=0) == 1
    assert list(db.RECORDINGS.glob("*.wav")) == []
    assert db.history_list()[0]["audio_path"] is None
    assert db.history_list()[0]["text"] == "only text"


def test_prune_sweeps_old_orphan_files(db):
    """A crash between writing the wav and inserting the row leaves a file
    nothing points at. Once it's older than the grace window, it goes."""
    import time

    _wav(db, "orphan.wav")
    db.history_add("dictation", "kept", 1.0, audio_path=_wav(db, "kept.wav"))
    later = time.time() + db.ORPHAN_GRACE_SECS + 1
    assert db.prune_recordings(keep=3, now=later) == 1
    assert [p.name for p in db.RECORDINGS.glob("*.wav")] == ["kept.wav"]


def test_prune_does_not_delete_a_dictation_still_being_saved(db):
    """The race: _save_audio() writes the .wav, then history_add() inserts the
    row. In between, nothing references the file — and it looks exactly like an
    orphan. A prune fired from a Settings save used to delete the recording of
    the dictation in progress, leaving the new row pointing at a dead path."""
    in_flight = _wav(db, "in-flight.wav")          # written, row not yet inserted
    db.prune_recordings(keep=3)                    # concurrent Settings save

    assert Path(in_flight).exists(), "in-flight recording was swept as an orphan"

    db.history_add("dictation", "saved", 1.0, audio_path=in_flight)   # row lands
    row = db.history_list(limit=1)[0]
    assert Path(row["audio_path"]).exists()


def test_superseded_files_are_deleted_regardless_of_age(db):
    """A file whose row we just cleared is known-dead. The grace window applies
    only to files nothing ever referenced, not to these."""
    for i in range(4):
        db.history_add("dictation", f"c{i}", 1.0, audio_path=_wav(db, f"s{i}.wav"))
    assert db.prune_recordings(keep=1) == 3        # fresh, but superseded
    assert len(list(db.RECORDINGS.glob("*.wav"))) == 1


def test_prune_never_unlinks_a_survivors_shared_file(db):
    """Databases written before the filename-collision fix can have two rows
    pointing at one file. Pruning the older row must not delete the newer
    row's audio."""
    shared = _wav(db, "shared.wav")
    db.history_add("dictation", "older", 1.0, audio_path=shared)
    db.history_add("dictation", "newer", 1.0, audio_path=shared)
    db.prune_recordings(keep=1)
    assert (db.RECORDINGS / "shared.wav").exists()


def test_readback_rows_are_untouched(db):
    db.history_add("readback", "narrated", 1.0)          # no audio_path
    for i in range(4):
        db.history_add("dictation", f"c{i}", 1.0, audio_path=_wav(db, f"x{i}.wav"))
    db.prune_recordings(keep=1)
    assert len(db.history_list()) == 5


def test_lowering_the_cap_prunes_immediately(client, db):
    for i in range(4):
        db.history_add("dictation", f"c{i}", 1.0, audio_path=_wav(db, f"y{i}.wav"))
    assert client.put("/api/settings", json={"recordings_keep": "1"}).status_code == 200
    assert len(list(db.RECORDINGS.glob("*.wav"))) == 1


def test_recordings_keep_is_validated(db):
    db.validate_setting("recordings_keep", "0")
    db.validate_setting("recordings_keep", "3")
    with pytest.raises(ValueError):
        db.validate_setting("recordings_keep", "-1")
    with pytest.raises(ValueError):
        db.validate_setting("recordings_keep", "lots")


# -------------------------------------------------------------------- server

def test_bad_input_is_400_not_500(client):
    assert client.get("/api/history?limit=abc").status_code == 400
    assert client.put("/api/snippets/1", json={"expansion": "x"}).status_code == 400
    assert client.put("/api/settings", json={"tts_rate": "fast"}).status_code == 400


def test_rejected_hotkey_does_not_persist(client, db):
    before = db.get("hotkey_speak")
    assert client.put("/api/settings", json={"hotkey_speak": "garbage"}).status_code == 400
    assert db.get("hotkey_speak") == before        # written only after validating
    import server
    assert server.HOTKEYS.listener is not None     # listener survived


def test_valid_settings_round_trip(client, db):
    assert client.put("/api/settings", json={"tts_rate": "240"}).status_code == 200
    assert db.get("tts_rate") == "240"


def test_non_loopback_host_is_refused(client):
    """Any web page can POST to 127.0.0.1, and DNS rebinding can point an
    attacker's domain here. A Host check defeats both."""
    assert client.post("/api/speech/stop",
                       headers={"Host": "evil.example.com"}).status_code == 403
    assert client.post("/api/speech/stop",
                       headers={"Origin": "https://evil.example.com"}).status_code == 403
    assert client.post("/api/speech/stop").status_code == 200


@pytest.mark.parametrize("origin", [
    "http://localhost.evil.example",      # passed the old prefix check
    "http://127.0.0.1.evil.example",      # passed the old prefix check
    "http://localhost:9999",              # another local server, wrong port
    "https://localhost",                  # wrong scheme
    "null",
])
def test_lookalike_origin_is_refused(client, db, origin):
    """Origin must match scheme, host and port exactly, not by prefix."""
    before = db.get("snippets_enabled")
    r = client.post("/api/snippets/toggle", headers={"Origin": origin})
    assert r.status_code == 403
    assert db.get("snippets_enabled") == before          # nothing written


def test_dashboard_own_origin_is_allowed(client):
    # the test client serves on localhost:80, so that is the server's own origin
    for origin in ("http://localhost", "http://127.0.0.1", "http://[::1]"):
        assert client.post("/api/speech/stop",
                           headers={"Origin": origin}).status_code == 200
    import server
    assert "http://127.0.0.1:8765" in server._allowed_origins("8765")
    assert "http://127.0.0.1:8766" not in server._allowed_origins("8765")


def test_system_endpoint_reports_hardware(client):
    d = client.get("/api/system").get_json()
    assert d["backend"] in ("mlx", "faster-whisper")
    assert len(d["models"]) == 6
    assert sum(m["recommended"] for m in d["models"]) == 1


def test_status_endpoint_reports_permissions_and_model(client):
    d = client.get("/api/status").get_json()
    assert set(d) >= {"accessibility", "model_state", "model"}
    assert d["model_state"] in ("loading", "ready", "error", "unknown")


def test_status_surfaces_missing_accessibility(client, monkeypatch):
    """The modal first-run failure: hotkeys register but never fire. The app
    must say so rather than printing 'Hotkeys active'."""
    import permissions
    import server

    monkeypatch.setattr(server.permissions, "summary", lambda: {
        "accessibility": False, "hint": "grant it"})
    d = client.get("/api/status").get_json()
    assert d["accessibility"] is False
    assert d["accessibility_hint"] == "grant it"


def test_hotkeys_active_is_not_logged_without_permission(db, capsys, monkeypatch):
    import core

    monkeypatch.setattr(core.permissions, "accessibility_trusted", lambda: False)
    hk = core.HotkeyManager(core.Core())
    hk.start()
    out = capsys.readouterr().out
    assert "Hotkeys active" not in out          # the old lie
    assert "will NOT fire" in out               # the new truth
    hk.stop()


def test_default_dictate_hotkey_avoids_macos_input_switcher(db):
    """⌃⌥Space is 'Select next source in Input menu' for anyone with two
    keyboard layouts."""
    assert db.DEFAULTS["hotkey_dictate"] != "<ctrl>+<alt>+<space>"
    db.validate_setting("hotkey_dictate", db.DEFAULTS["hotkey_dictate"])


def test_model_state_transitions(db, monkeypatch):
    import core

    c = core.Core()
    assert c.model_state == "loading"
    monkeypatch.setattr(core, "transcribe", lambda *a, **k: "")
    c.warm_up()
    assert c.model_state == "ready"


def test_model_failure_is_recorded_not_swallowed(db, monkeypatch):
    import core

    c = core.Core()

    def boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(core, "transcribe", boom)
    with pytest.raises(RuntimeError):
        c.warm_up()
    assert c.model_state == "error"
    assert "no network" in c.model_error


def test_port_in_use_detection():
    import socket

    import server
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.listen(1)
    assert server.port_in_use(port) is True
    s.close()
    assert server.port_in_use(port) is False


# ------------------------------------------------------- microphone recovery

class _FakeStream:
    def __init__(self, fail_start=False):
        self.fail_start = fail_start
        self.closed = False
        self.started = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("Error opening InputStream: PaErrorCode -9986")
        self.started = True

    def stop(self): pass

    def close(self): self.closed = True


class _FakeSD:
    """Mimics sounddevice: opens fail until PortAudio is reinitialised."""

    def __init__(self, heal_on_reinit=True):
        self.reinitialised = False
        self.heal_on_reinit = heal_on_reinit
        self.streams = []

    def InputStream(self, **kw):
        broken = not (self.reinitialised and self.heal_on_reinit)
        s = _FakeStream(fail_start=broken)
        self.streams.append(s)
        return s

    def _terminate(self): pass

    def _initialize(self): self.reinitialised = True


def _install_fake_sd(monkeypatch, fake):
    import sys
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_stale_audio_devices_are_rescanned_and_recording_recovers(db, monkeypatch):
    """PortAudio caches the device list at init. Plug in headphones while
    Walnut runs and every mic open fails with a bare internal error — the
    process is dead to dictation until restart. It must rescan and retry."""
    import core

    fake = _FakeSD(heal_on_reinit=True)
    _install_fake_sd(monkeypatch, fake)
    c = core.Core()
    c._start_recording()

    assert fake.reinitialised, "did not rescan devices after the first failure"
    assert c.recording(), "should be recording after the retry succeeded"
    assert fake.streams[0].closed, "the failed stream must not be leaked"
    c.stream = None


def test_a_truly_dead_microphone_fails_cleanly(db, monkeypatch):
    """If it fails even after a rescan: no exception, no phantom stream, and
    recording() must not lie."""
    import core

    fake = _FakeSD(heal_on_reinit=False)
    _install_fake_sd(monkeypatch, fake)
    c = core.Core()
    states = []
    c.on_state = states.append
    c._start_recording()

    assert c.stream is None and not c.recording()
    assert states[-1] == "idle"
    assert all(s.closed for s in fake.streams), "half-open streams left behind"
