"""Regression tests. Every test here is a bug Walnut actually shipped.

Run with:  uv run --group dev pytest -q
"""

import os
import time
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

    _wav(db, "dictation-20200101-000000.wav")
    db.history_add("dictation", "kept", 1.0, audio_path=_wav(db, "dictation-kept.wav"))
    later = time.time() + db.ORPHAN_GRACE_SECS + 1
    assert db.prune_recordings(keep=3, now=later) == 1
    assert [p.name for p in db.RECORDINGS.glob("*.wav")] == ["dictation-kept.wav"]


def test_sweep_leaves_files_that_are_not_ours(db):
    import time

    mine = _wav(db, "dictation-20200101-000000.wav")
    theirs = _wav(db, "my-voice-memo.wav")
    later = time.time() + db.ORPHAN_GRACE_SECS + 1
    assert db.prune_recordings(keep=3, now=later) == 1
    assert not Path(mine).exists() and Path(theirs).exists()


def test_history_delete_only_unlinks_inside_recordings(db, tmp_path):
    outside = tmp_path / "precious.wav"
    outside.write_bytes(b"RIFF")
    sneaky = str(db.RECORDINGS / ".." / "precious.wav")   # resolves outside
    inside = _wav(db, "dictation-x.wav")
    a = db.history_add("dictation", "a", 1.0, audio_path=str(outside))
    b = db.history_add("dictation", "b", 1.0, audio_path=sneaky)
    c = db.history_add("dictation", "c", 1.0, audio_path=inside)
    for hid in (a, b, c):
        db.history_delete(hid)
    assert outside.exists()
    assert not Path(inside).exists()
    assert db.history_get(a) is None and db.history_get(c) is None


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


def test_origin_must_be_exactly_loopback(client):
    """startswith() let http://localhost.evil.com and http://127.0.0.1.evil.com
    through. The hostname must match exactly; the port may be anything."""
    for bad in ("http://localhost.evil.com", "http://127.0.0.1.evil.com",
                "http://localhost.evil.com:8080", "https://evil.com"):
        r = client.post("/api/speech/stop", headers={"Origin": bad})
        assert r.status_code == 403, bad
    for good in ("http://localhost:5000", "http://127.0.0.1:5000",
                 "http://[::1]:5000", "http://localhost"):
        r = client.post("/api/speech/stop", headers={"Origin": good})
        assert r.status_code == 200, good


def test_failed_command_start_does_not_hijack_next_dictation(db, monkeypatch):
    """If the Console-command hotkey cannot start recording, command_next must
    not stay set, or the next plain dictation is sent to the Console."""
    import core

    monkeypatch.setattr(core, "play_sound", lambda *a, **k: None)
    fake = _FakeSD(heal_on_reinit=False)
    _install_fake_sd(monkeypatch, fake)
    c = core.Core()
    c.on_state = lambda s: None

    c.toggle_command()                       # mic open fails (both tries)
    assert not c.recording()
    assert c.command_next is False

    # a raised error on the start path must clear it too
    def boom():
        raise RuntimeError("no audio stack")
    monkeypatch.setattr(c, "_start_recording", boom)
    try:
        c.toggle_command()
    except RuntimeError:
        pass
    assert c.command_next is False
    assert not c.lock.locked()


# ------------------------------------------------------------ clipboard (m1)

class _FakeData:  # stands in for NSData
    def __init__(self, b): self.b = b


class _FakeItem:
    def __init__(self, d=None): self.d = dict(d or {})
    @classmethod
    def alloc(cls): return cls()
    def init(self): return self
    def types(self): return list(self.d)
    def dataForType_(self, t): return self.d.get(t)
    def setData_forType_(self, data, t): self.d[t] = data; return True


class _FakePasteboard:
    def __init__(self, items): self.items = items
    def pasteboardItems(self): return self.items
    def clearContents(self): self.items = []
    def writeObjects_(self, objs): self.items = list(objs)


def _fake_appkit(monkeypatch, pb):
    import sys
    import types
    mod = types.ModuleType("AppKit")
    mod.NSPasteboard = types.SimpleNamespace(generalPasteboard=lambda: pb)
    mod.NSPasteboardItem = _FakeItem
    monkeypatch.setitem(sys.modules, "AppKit", mod)


def test_clipboard_image_survives_dictation_paste(db, monkeypatch):
    import core
    png = _FakeData(b"\x89PNG")
    pb = _FakePasteboard([_FakeItem({"public.png": png, "public.tiff": _FakeData(b"t")})])
    _fake_appkit(monkeypatch, pb)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    # pbcopy stand-in: replaces the clipboard with plain text
    monkeypatch.setattr(core, "set_clipboard", lambda t: setattr(
        pb, "items", [_FakeItem({"public.utf8-plain-text": _FakeData(t.encode())})]))
    c = core.Core()
    c._type("hello")
    assert len(pb.items) == 1
    assert pb.items[0].d["public.png"] is png
    assert set(pb.items[0].d) == {"public.png", "public.tiff"}


def test_empty_clipboard_restores_to_empty(db, monkeypatch):
    import core
    pb = _FakePasteboard([])
    _fake_appkit(monkeypatch, pb)
    snap = core.save_clipboard()
    pb.items = [_FakeItem({"public.utf8-plain-text": _FakeData(b"x")})]
    core.restore_clipboard(snap)
    assert pb.items == []


def test_clipboard_without_appkit_falls_back_to_text(monkeypatch):
    import sys
    import core
    monkeypatch.setitem(sys.modules, "AppKit", None)   # import raises
    sent = []
    monkeypatch.setattr(core, "get_clipboard", lambda: "old")
    monkeypatch.setattr(core, "set_clipboard", sent.append)
    core.restore_clipboard(core.save_clipboard())
    assert sent == ["old"]


# ------------------------------------------------------------ races (m2)

def test_clipboard_users_do_not_interleave(db, monkeypatch):
    """A paste and a selection copy running together must take turns."""
    import threading
    import core
    events = []
    real_sleep = time.sleep
    monkeypatch.setattr(core.time, "sleep", lambda s: real_sleep(0.01))
    monkeypatch.setattr(core, "save_clipboard", lambda: events.append("save") or "snap")
    monkeypatch.setattr(core, "restore_clipboard", lambda s: events.append("restore"))
    monkeypatch.setattr(core, "set_clipboard", lambda t: events.append("set"))
    monkeypatch.setattr(core, "get_clipboard", lambda: events.append("get") or "x")
    c = core.Core()
    import contextlib
    c.kb = type("KB", (), {"pressed": lambda s, k: contextlib.nullcontext(),
                           "press": lambda s, k: None,
                           "release": lambda s, k: None})()
    ts = [threading.Thread(target=c._type, args=("t",)),
          threading.Thread(target=c._copy_selection),
          threading.Thread(target=c._type, args=("u",))]
    for t in ts: t.start()
    for t in ts: t.join(10)
    # each user's save ... restore is a contiguous block
    flat = " ".join(events).replace("save", "|save").split("|")[1:]
    assert len(flat) == 3
    assert all(b.strip().startswith("save") and b.strip().endswith("restore") and
               b.count("save") == 1 for b in flat), events


def test_concurrent_speak_leaves_one_cuttable_say(db, monkeypatch):
    import threading
    import core

    procs = []

    class FakeProc:
        def __init__(self, *a, **k):
            time.sleep(0.02)          # widen the race window
            self.dead = False
            self.stdin = type("S", (), {"write": lambda s, b: None,
                                        "close": lambda s: None})()
            procs.append(self)
        def poll(self): return 1 if self.dead else None
        def terminate(self): self.dead = True

    monkeypatch.setattr(core.subprocess, "Popen", FakeProc)
    c = core.Core()
    ts = [threading.Thread(target=c.speak, args=("hi", False)) for _ in range(6)]
    for t in ts: t.start()
    for t in ts: t.join(10)
    assert len(procs) == 6
    c.stop_all()
    assert all(p.dead for p in procs), "a say process was orphaned"


# ------------------------------------------------------------ console url (m4)

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:4173", "http://localhost:4173/", "http://[::1]:4173",
    "http://127.5.5.5", "https://console.example.com", "https://10.0.0.5:8443"])
def test_console_url_allowed(db, url):
    assert db.validate_setting("console_url", url) == url


@pytest.mark.parametrize("url", [
    "http://example.com", "http://192.168.1.20:4173", "http://10.0.0.5",
    "http://localhost.evil.com", "http://127.0.0.1.evil.com", "http://",
    "ftp://localhost"])
def test_console_url_refused(db, url):
    with pytest.raises(ValueError):
        db.validate_setting("console_url", url)


def test_console_url_refusal_is_a_400_with_a_clear_message(client):
    r = client.put("/api/settings", json={"console_url": "http://example.com"})
    assert r.status_code == 400
    assert "https" in r.get_json()["error"]


# ------------------------------------------------------------ bad bodies (m6)

@pytest.mark.parametrize("method,url,body", [
    ("post", "/api/words", {"word": 5}),
    ("post", "/api/words", [1, 2]),
    ("post", "/api/replacements", {"wrong": 1, "right": "x"}),
    ("post", "/api/replacements", ["a"]),
    ("post", "/api/snippets", {"trigger": ["a"], "expansion": "x"}),
    ("post", "/api/snippets", ["a"]),
    ("put", "/api/snippets/1", ["a"]),
    ("put", "/api/settings", ["a"]),
    ("put", "/api/settings", {"tts_rate": "inf"}),
    ("put", "/api/settings", {"tts_rate": 1e999}),
    ("post", "/api/test-voice", {"text": 5}),
])
def test_bad_bodies_are_400_not_500(client, method, url, body):
    assert getattr(client, method)(url, json=body).status_code == 400


def test_good_bodies_still_work(client, db):
    assert client.post("/api/words", json={"word": "Walnut"}).status_code == 200
    assert client.post("/api/words").status_code in (200, 415)   # unchanged: no body
    assert [w["word"] for w in db.words_list()] == ["Walnut"]


# ------------------------------------------------------------ stt_model (m7)

def test_unknown_stt_model_is_rejected(client, db):
    before = db.get("stt_model")
    r = client.put("/api/settings", json={"stt_model": "large-v3-trubo"})
    assert r.status_code == 400
    assert "unknown speech model" in r.get_json()["error"]
    assert db.get("stt_model") == before


def test_known_and_legacy_stt_model_names_still_work(db):
    import stt
    for m in stt._BY_ID:
        assert db.validate_setting("stt_model", m) == m
    for old, new in stt._LEGACY.items():
        assert db.validate_setting("stt_model", old) == stt.canonical(old)
    assert db.validate_setting("stt_model", "") == stt.default_model()


# ------------------------------------------------------------ console bridge (m11)

def test_console_bridge_posts_to_the_configured_url(db, monkeypatch):
    import io
    import json
    import console_bridge

    db.set_setting("console_url", "https://console.example.com/")
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return io.BytesIO(json.dumps({"id": None, "target": "vet-skill"}).encode())

    monkeypatch.setattr(console_bridge.urllib.request, "urlopen", fake_urlopen)
    spoken = []
    console_bridge.handle("  run the vet skill. ", spoken.append)
    assert seen[0].full_url == "https://console.example.com/api/run"
    assert seen[0].get_method() == "POST"
    assert json.loads(seen[0].data) == {"input": "run the vet skill"}
    assert spoken == ["Running vet skill."]
