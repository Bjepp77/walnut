"""SQLite storage for Walnut: settings, dictionary, snippets, history, stats."""

import contextlib
import ipaddress
import re
import sqlite3
import time
import tomllib
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path

import stt as stt_backend

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "walnut.db"
RECORDINGS = HERE / "recordings"

DEFAULTS = {
    "tts_voice": "",
    "tts_rate": "210",
    # Picked for the Mac this is running on: the big model where the GPU makes
    # it cheap, a small one where transcription lands on the CPU.
    "stt_model": stt_backend.default_model(),
    "stt_backend": "auto",   # 'auto' | 'mlx' | 'faster-whisper'
    "stt_language": "en",
    "typing_wpm": "60",
    # How many dictation .wav files to keep on disk. Transcripts are kept
    # forever and cost nothing; the audio is what fills the drive. 0 = none.
    "recordings_keep": "3",
    "hotkey_speak": "<ctrl>+<alt>+s",
    # NOT ⌃⌥Space: macOS binds that to "Select next source in Input menu" for
    # anyone with two keyboard layouts. Existing databases keep their own value.
    "hotkey_dictate": "<ctrl>+<alt>+d",
    "hotkey_command": "<ctrl>+<alt>+c",
    "console_url": "http://127.0.0.1:4173",
    "snippets_enabled": "1",
    "port": "8765",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY, word TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS replacements (
    id INTEGER PRIMARY KEY, wrong TEXT NOT NULL, right TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snippets (
    id INTEGER PRIMARY KEY, trigger TEXT NOT NULL, expansion TEXT NOT NULL,
    uses INTEGER DEFAULT 0, enabled INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,             -- 'dictation' | 'readback'
    ts REAL NOT NULL,               -- unix seconds
    text TEXT NOT NULL,
    words INTEGER NOT NULL,
    duration REAL NOT NULL,         -- seconds
    snippet INTEGER DEFAULT 0,      -- 1 if a snippet expanded
    audio_path TEXT);
-- every history query orders or filters on ts
CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts);
CREATE INDEX IF NOT EXISTS idx_history_type_ts ON history(type, ts);
"""


@contextlib.contextmanager
def _conn():
    """A connection that is always committed-or-rolled-back, and always closed.

    The previous version returned a bare Connection used as `with _conn() as c`.
    sqlite3's context manager commits the transaction but does NOT close the
    connection, so every call leaked a file descriptor until the GC happened to
    collect it — a few hundred dashboard polls could exhaust the fd limit.
    """
    c = sqlite3.connect(DB_PATH, timeout=10.0)
    c.row_factory = sqlite3.Row
    try:
        with c:
            yield c
    finally:
        c.close()


def init() -> None:
    RECORDINGS.mkdir(exist_ok=True)
    with _conn() as c:
        # WAL lets the Flask threads read while the dictation thread writes,
        # instead of serialising on a whole-database lock. Persists on the file.
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.executescript(SCHEMA)
        fresh = c.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0
    if fresh:
        _seed_from_config()


def _seed_from_config() -> None:
    """First run: import the old config.toml so nothing is lost."""
    cfg = {}
    cfg_path = HERE / "config.toml"
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    settings = dict(DEFAULTS)
    # NB: locals are suffixed to avoid shadowing the `stt` speech module.
    tts_c, stt_c, hot_c = (cfg.get("tts", {}), cfg.get("stt", {}),
                           cfg.get("hotkeys", {}))
    if tts_c.get("voice") is not None:
        settings["tts_voice"] = str(tts_c.get("voice", ""))
    if tts_c.get("rate"):
        settings["tts_rate"] = str(tts_c["rate"])
    # An empty model in config.toml means "let Walnut choose for this Mac".
    if stt_c.get("model"):
        settings["stt_model"] = stt_backend.canonical(stt_c["model"])
    if stt_c.get("backend"):
        settings["stt_backend"] = stt_c["backend"]
    if stt_c.get("language"):
        settings["stt_language"] = stt_c["language"]
    if hot_c.get("speak"):
        settings["hotkey_speak"] = hot_c["speak"]
    if hot_c.get("dictate"):
        settings["hotkey_dictate"] = hot_c["dictate"]
    with _conn() as c:
        c.executemany("INSERT OR REPLACE INTO settings VALUES (?,?)",
                      settings.items())
        for w in cfg.get("vocabulary", {}).get("hints", []):
            c.execute("INSERT OR IGNORE INTO words(word) VALUES (?)", (w,))
        for wrong, right in cfg.get("replacements", {}).items():
            c.execute("INSERT INTO replacements(wrong, right) VALUES (?,?)",
                      (wrong, right))
        for trig, exp in cfg.get("snippets", {}).items():
            c.execute("INSERT INTO snippets(trigger, expansion) VALUES (?,?)",
                      (trig, exp))


# ---------------------------------------------------------------- settings
#
# Everything here is reachable from the dashboard, so a bad value must be
# rejected at the door. It used to be written first and interpreted later: a
# non-numeric tts_rate crashed narration, and an unparseable hotkey killed the
# listener and then crash-looped the app at next launch (KeepAlive respawns it).

def _int_in(lo: int, hi: int):
    def check(v: str) -> str:
        try:
            n = int(float(v))     # accept "210" and "210.0"
        except OverflowError:     # "inf", "1e999"
            raise ValueError(f"must be between {lo} and {hi}")
        if not lo <= n <= hi:
            raise ValueError(f"must be between {lo} and {hi}")
        return str(n)
    return check


def _one_of(*allowed: str):
    def check(v: str) -> str:
        if v not in allowed:
            raise ValueError(f"must be one of {', '.join(allowed)}")
        return v
    return check


def _language(v: str) -> str:
    v = v.strip().lower()
    if v and not re.fullmatch(r"[a-z]{2,3}", v):
        raise ValueError("must be a language code like 'en', or blank for auto")
    return v


def _hotkey(v: str) -> str:
    from pynput.keyboard import HotKey   # lazy: store is imported by --doctor
    HotKey.parse(v)                      # raises ValueError on anything invalid
    return v


def _url(v: str) -> str:
    v = v.strip()
    if not v.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) URL: {v!r}")
    # Spoken commands are POSTed here. Plain http to another machine would send
    # them unencrypted, so only https (anywhere) or http to this Mac is allowed.
    try:
        host = urllib.parse.urlsplit(v).hostname
    except ValueError:
        raise ValueError(f"not a valid URL: {v!r}")
    if not host:
        raise ValueError(f"URL has no host: {v!r}")
    if v.startswith("http://") and not _is_loopback(host):
        raise ValueError(
            "plain http is only allowed to this Mac (localhost or 127.0.0.1); "
            "use an https:// address for another machine")
    return v


def _is_loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _stt_model(v: str) -> str:
    v = v.strip()
    if v and not stt_backend.is_known(v):     # blank = let Walnut choose
        raise ValueError(f"unknown speech model {v!r}")
    return stt_backend.canonical(v)


def _voice(v: str) -> str:
    if len(v) > 100:
        raise ValueError("voice name too long")
    return v


VALIDATORS = {
    "tts_rate": _int_in(80, 500),
    "typing_wpm": _int_in(5, 300),
    "recordings_keep": _int_in(0, 500),
    "port": _int_in(1024, 65535),
    "stt_model": _stt_model,
    "stt_backend": _one_of("auto", stt_backend.MLX, stt_backend.FASTER),
    "stt_language": _language,
    "snippets_enabled": _one_of("0", "1"),
    "hotkey_speak": _hotkey,
    "hotkey_dictate": _hotkey,
    "hotkey_command": _hotkey,
    "console_url": _url,
    "tts_voice": _voice,
}


def validate_setting(key: str, value) -> str:
    """Normalise a setting, or raise ValueError. Unknown keys are rejected."""
    if key not in DEFAULTS:
        raise ValueError(f"unknown setting {key!r}")
    return VALIDATORS[key](str(value))


def get(key: str) -> str:
    with _conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else DEFAULTS.get(key, "")


def get_valid(key: str) -> str:
    """get(), but a corrupt/legacy stored value falls back to the default.

    Read paths that would otherwise crash the app use this: a database carried
    over from an older Walnut (or hand-edited) must never take the process down.
    """
    raw = get(key)
    try:
        return validate_setting(key, raw)
    except Exception:
        default = DEFAULTS.get(key, "")
        print(f"[store] ignoring invalid {key}={raw!r}; using {default!r}",
              flush=True)
        return default


def set_setting(key: str, value: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, str(value)))


def all_settings() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT key, value FROM settings").fetchall()
    out = dict(DEFAULTS)
    out.update({r["key"]: r["value"] for r in rows})
    return out


# ---------------------------------------------------------------- dictionary

def words_list() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM words ORDER BY word")]


def words_add(word: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO words(word) VALUES (?)", (word.strip(),))


def words_delete(wid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM words WHERE id=?", (wid,))


def replacements_list() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM replacements ORDER BY wrong")]


def replacements_add(wrong: str, right: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO replacements(wrong, right) VALUES (?,?)",
                  (wrong.strip(), right.strip()))


def replacements_delete(rid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM replacements WHERE id=?", (rid,))


# ---------------------------------------------------------------- snippets

def snippets_list() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM snippets ORDER BY trigger")]


def snippets_add(trigger: str, expansion: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO snippets(trigger, expansion) VALUES (?,?)",
                  (trigger.strip(), expansion))


def snippets_update(sid: int, trigger: str, expansion: str, enabled: int) -> None:
    with _conn() as c:
        c.execute("UPDATE snippets SET trigger=?, expansion=?, enabled=? WHERE id=?",
                  (trigger.strip(), expansion, enabled, sid))


def snippets_delete(sid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM snippets WHERE id=?", (sid,))


def snippets_hit(sid: int) -> None:
    with _conn() as c:
        c.execute("UPDATE snippets SET uses = uses + 1 WHERE id=?", (sid,))


# ---------------------------------------------------------------- history

def history_add(kind: str, text: str, duration: float,
                snippet: bool = False, audio_path: str | None = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO history(type, ts, text, words, duration, snippet, audio_path)"
            " VALUES (?,?,?,?,?,?,?)",
            (kind, time.time(), text, len(text.split()), duration,
             int(snippet), audio_path))
        return cur.lastrowid


def history_list(q: str = "", kind: str = "", limit: int = 200) -> list[dict]:
    limit = max(1, min(int(limit), 1000))   # never let a caller ask for it all
    sql, args = "SELECT * FROM history WHERE 1=1", []
    if q:
        # escape LIKE wildcards so searching for "100%" isn't a match-anything
        q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += " AND text LIKE ? ESCAPE '\\'"
        args.append(f"%{q}%")
    if kind in ("dictation", "readback"):
        sql += " AND type=?"
        args.append(kind)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(sql, args)]


def history_get(hid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM history WHERE id=?", (hid,)).fetchone()
    return dict(row) if row else None


def _inside_recordings(path) -> bool:
    """True only for a path that really lives under the recordings folder."""
    try:
        return Path(path).resolve().is_relative_to(RECORDINGS.resolve())
    except (OSError, ValueError):
        return False


def history_delete(hid: int) -> None:
    entry = history_get(hid)
    # A row is data, not a licence to delete: only unlink our own recordings.
    if entry and entry.get("audio_path") and _inside_recordings(entry["audio_path"]):
        Path(entry["audio_path"]).unlink(missing_ok=True)
    with _conn() as c:
        c.execute("DELETE FROM history WHERE id=?", (hid,))


# A .wav is written by _save_audio() a moment before its history row is
# inserted. In that window nothing references it, so an orphan sweep would
# happily delete the recording of the dictation currently in progress. Give an
# unreferenced file this long to acquire its row before treating it as debris.
ORPHAN_GRACE_SECS = 60


def prune_recordings(keep: int | None = None, now: float | None = None) -> int:
    """Keep only the newest `keep` dictation recordings on disk.

    Transcripts are never touched — only the .wav files, which are the thing
    that grows without bound. A pruned entry keeps its history row; Replay just
    falls back to reading the text aloud instead of playing the original audio.

    Two kinds of file get deleted, and they are not the same kind of certain:

    * **Superseded** — a row pointed at it and we just cleared that row. Known
      dead. Deleted regardless of age.
    * **Orphaned** — nothing in the database mentions it. Usually a crash
      leftover, but it is *also* what an in-flight dictation looks like for the
      microseconds between writing the file and inserting its row. Deleted only
      once it is older than ORPHAN_GRACE_SECS.

    `now` is injectable so the grace window can be tested without sleeping.

    Returns the number of files deleted.
    """
    if keep is None:
        keep = int(get_valid("recordings_keep"))
    keep = max(0, keep)
    now = time.time() if now is None else now

    with _conn() as c:
        rows = c.execute(
            "SELECT id, audio_path FROM history WHERE audio_path IS NOT NULL"
            " ORDER BY ts DESC, id DESC").fetchall()
        survivors = {r["audio_path"] for r in rows[:keep]}
        superseded = {r["audio_path"] for r in rows[keep:]}
        for r in rows[keep:]:
            c.execute("UPDATE history SET audio_path=NULL WHERE id=?", (r["id"],))

    # Resolve first: a pre-fix database can hold two rows pointing at the same
    # file, and a survivor's path must never be unlinked as another's leftover.
    protected = {Path(p).resolve() for p in survivors}
    removed = 0

    for p in superseded:
        target = Path(p).resolve()
        if target not in protected:
            target.unlink(missing_ok=True)
            removed += 1

    if not RECORDINGS.exists():
        return removed

    known = protected | {Path(p).resolve() for p in superseded}
    # Only files named the way _save_audio() names them: anything else in this
    # folder was put there by the user and is not ours to sweep.
    for f in RECORDINGS.glob("dictation-*.wav"):
        if f.resolve() in known:
            continue
        try:
            age = now - f.stat().st_mtime
        except OSError:          # vanished under us; nothing to do
            continue
        if age < ORPHAN_GRACE_SECS:
            continue             # a dictation may be mid-write. Leave it.
        f.unlink(missing_ok=True)
        removed += 1
    return removed


# ---------------------------------------------------------------- stats

def _day(ts: float) -> date:
    return datetime.fromtimestamp(ts).date()


def stats(period: str = "week") -> dict:
    days = {"week": 7, "month": 30, "all": 3650}[period]
    cutoff = time.time() - days * 86400
    typing_wpm = max(1, int(float(get_valid("typing_wpm"))))

    with _conn() as c:
        # Only the columns the maths needs. This used to `SELECT *`, dragging
        # every transcript ever recorded into memory on each dashboard poll;
        # `keystrokes` needs the text length, not the text.
        rows = [dict(r) for r in c.execute(
            "SELECT type, ts, words, duration, LENGTH(text) AS chars"
            " FROM history ORDER BY ts")]
        snippet_uses = c.execute(
            "SELECT COALESCE(SUM(uses),0) FROM snippets").fetchone()[0]

    cur = [r for r in rows if r["ts"] >= cutoff]
    dic = [r for r in cur if r["type"] == "dictation"]
    rb = [r for r in cur if r["type"] == "readback"]

    words_dictated = sum(r["words"] for r in dic)
    speak_secs = sum(r["duration"] for r in dic)
    keystrokes = sum(r["chars"] for r in dic)
    focus_secs = sum(r["duration"] for r in cur)
    # time saved = what typing those words would have cost, minus speaking time
    saved = max(0.0, words_dictated / typing_wpm * 60 - speak_secs)
    speaking_wpm = round(words_dictated / (speak_secs / 60)) if speak_secs else 0

    # streaks over all history
    active = sorted({_day(r["ts"]) for r in rows})
    streak, longest, run, prev = 0, 0, 0, None
    for d in active:
        run = run + 1 if prev and (d - prev).days == 1 else 1
        longest = max(longest, run)
        prev = d
    today = date.today()
    d, sset = today, set(active)
    if d not in sset:
        d -= timedelta(days=1)
    while d in sset:
        streak += 1
        d -= timedelta(days=1)

    # daily activity for charts (chart window capped at 30 days)
    chart_days = min(days, 30)
    labels, dic_series, rb_series = [], [], []
    for i in range(chart_days - 1, -1, -1):
        d = today - timedelta(days=i)
        labels.append(d.strftime("%b %-d"))
        dic_series.append(sum(1 for r in dic if _day(r["ts"]) == d))
        rb_series.append(sum(1 for r in rb if _day(r["ts"]) == d))

    period_days = min(days, max(1, (today - active[0]).days + 1) if active else 1)
    active_in_period = len({_day(r["ts"]) for r in cur})
    momentum = round(100 * active_in_period / period_days)

    return {
        "streak": streak,
        "longest_streak": longest,
        "active_days": len(active),
        "speaking_wpm": speaking_wpm,
        "typing_wpm": typing_wpm,
        "time_saved_secs": round(saved),
        "momentum": momentum,
        "readback_sessions": len(rb),
        "dictation_sessions": len(dic),
        "words_dictated": words_dictated,
        "keystrokes_saved": keystrokes,
        "snippet_uses": snippet_uses,
        "focus_secs": round(focus_secs),
        "chart": {"labels": labels, "dictation": dic_series, "readback": rb_series},
    }
