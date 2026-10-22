"""Watch the Tray Audio iCloud folder; transcribe new files; hand off to audio-digest.

  uv run watch.py            # poll forever
  uv run watch.py --once     # one pass, then exit
  uv run watch.py FILE       # process one file now

Skips iCloud placeholders (.name.icloud), hidden files, files still growing,
and anything already done (state file keyed by name+size+mtime).
Does NOT file reminders or events: it only runs `audio_digest.py chunk`; the
agent-run audio-digest skill does the rest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import file_mode

WATCH_DIR = Path("~/Library/Mobile Documents/com~apple~CloudDocs/Tray Audio").expanduser()
OUT_DIR = Path("~/MegaMind/Resources/raw/audio-transcripts").expanduser()
STATE = Path("~/.walnut/file_mode_done.json").expanduser()
DIGEST = Path("~/Work/tools/audio_digest.py").expanduser()
POLL_SECONDS = 15
STABLE_SECONDS = 20


def file_key(p: Path) -> str:
    st = p.stat()
    return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"


def load_done(state: Path | None = None) -> set[str]:
    state = state or STATE
    try:
        return set(json.loads(state.read_text()))
    except (OSError, ValueError):
        return set()


def save_done(done: set[str], state: Path | None = None) -> None:
    state = state or STATE
    state.parent.mkdir(parents=True, exist_ok=True)
    tmp = state.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(done)))
    os.replace(tmp, state)


def candidates(folder: Path | None = None) -> list[Path]:
    folder = folder or WATCH_DIR
    out = []
    for p in sorted(folder.iterdir()) if folder.is_dir() else []:
        if p.name.startswith(".") or p.name.endswith(".icloud"):
            continue  # hidden, or iCloud placeholder not downloaded yet
        if p.is_file() and p.suffix.lower() in file_mode.AUDIO_EXTS:
            out.append(p)
    return out


def is_stable(p: Path, seen: dict, now: float, stable: float = STABLE_SECONDS) -> bool:
    """True once size+mtime have not moved for `stable` seconds."""
    st = p.stat()
    sig = (st.st_size, st.st_mtime)
    prev = seen.get(p)
    if prev is None or prev[0] != sig:
        seen[p] = (sig, now)
        return False
    return now - prev[1] >= stable


def process(p: Path, out_dir: Path | None = None, run=None, handoff: bool = True) -> Path:
    out_dir = out_dir or OUT_DIR
    mtime = dt.datetime.fromtimestamp(p.stat().st_mtime)
    out = out_dir / f"{mtime:%Y-%m-%d_%H%M}_{file_mode.slugify(p.stem)}.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = file_mode.transcribe_chunks(file_mode.iter_audio_chunks(p), run=run)
    lines = file_mode.segments_to_lines(segs)
    body = file_mode.render_markdown(p.stem, f"{mtime:%Y-%m-%d %H:%M}", str(p), lines)
    tmp = out.with_suffix(".md.part")
    tmp.write_text(body)
    os.replace(tmp, out)  # never expose a half-written transcript
    if handoff and DIGEST.exists():
        r = subprocess.run(["python3", str(DIGEST), "chunk", str(out)],
                           capture_output=True, text=True)
        print(r.stdout)
        if r.returncode:
            print(r.stderr, file=sys.stderr)
    return out


def scan_once(seen: dict, done: set[str], **kw) -> list[Path]:
    made = []
    now = time.time()
    for p in candidates():
        try:
            key = file_key(p)
            if key in done or not is_stable(p, seen, now):
                continue
            made.append(process(p, **kw))
            done.add(key)
            save_done(done)
        except Exception as e:  # one bad file must not stop the watcher
            print(f"walnut watch: {p.name}: {e}", file=sys.stderr)
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.file:
        print(process(Path(a.file).expanduser()))
        return
    seen: dict = {}
    done = load_done()
    while True:
        scan_once(seen, done)
        if a.once:
            # --once still needs a stability window: second pass after STABLE_SECONDS
            time.sleep(STABLE_SECONDS + 1)
            scan_once(seen, done)
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
