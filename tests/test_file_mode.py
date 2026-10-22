"""File mode: chunked decode, timestamps, watcher rules. No model is loaded."""
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

import file_mode
import watch


def _wav(path, seconds, sr=16000):
    sf.write(str(path), (np.sin(np.arange(int(seconds * sr)) / 20) * 0.1).astype("float32"), sr)
    return path


def test_chunks_have_offsets_and_cover_all_audio(tmp_path):
    p = _wav(tmp_path / "a.wav", 25)
    chunks = list(file_mode.iter_audio_chunks(p, chunk_seconds=10))
    assert [o for o, _ in chunks] == [0.0, 10.0, 20.0]
    assert sum(len(a) for _, a in chunks) == 25 * 16000
    assert all(a.dtype == np.float32 and a.ndim == 1 for _, a in chunks)


def test_non_16k_wav_is_resampled_via_decoder(tmp_path):
    p = _wav(tmp_path / "b.wav", 5, sr=44100)
    total = sum(len(a) for _, a in file_mode.iter_audio_chunks(p, chunk_seconds=2))
    assert abs(total - 5 * 16000) < 1600


@pytest.mark.skipif(sys.platform != "darwin", reason="afconvert is macOS only")
def test_m4a_decodes(tmp_path):
    w = _wav(tmp_path / "c.wav", 6)
    m = tmp_path / "c.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(w), str(m)], check=True)
    total = sum(len(a) for _, a in file_mode.iter_audio_chunks(m, chunk_seconds=4))
    assert abs(total - 6 * 16000) < 3200


def test_segments_get_absolute_times_and_format(tmp_path):
    chunks = [(0.0, np.zeros(10, "float32")), (120.0, np.zeros(10, "float32"))]
    run = lambda a: [{"start": 1.0, "end": 2.0, "text": "hi"}, {"start": 3, "end": 4, "text": ""}]
    segs = list(file_mode.transcribe_chunks(chunks, run=run))
    assert [s["start"] for s in segs] == [1.0, 121.0]
    assert file_mode.segments_to_lines(segs) == ["[00:00:01] hi", "[00:02:01] hi"]
    assert file_mode.fmt_ts(3725) == "01:02:05"


def test_output_parses_with_audio_digest(tmp_path):
    md = file_mode.render_markdown("T", "2026-10-22 09:00", "/x/y.m4a", ["[00:00:05] hello"])
    assert md.startswith("Title: T\nDate: 2026-10-22 09:00\nSource: /x/y.m4a\n====\n")
    tool = os.path.expanduser("~/Work/tools/audio_digest.py")
    if os.path.exists(tool):
        f = tmp_path / "t.md"
        f.write_text(md)
        out = subprocess.run(["python3", tool, "chunk", str(f)], capture_output=True, text=True)
        assert out.returncode == 0 and "hello" in out.stdout


def test_watcher_skips_placeholders_hidden_and_non_audio(tmp_path):
    for n in ["a.m4a", ".b.m4a.icloud", ".hidden.wav", "notes.txt", "c.mp3"]:
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in watch.candidates(tmp_path)] == ["a.m4a", "c.mp3"]


def test_growing_file_not_stable_until_quiet(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"x")
    seen = {}
    assert not watch.is_stable(p, seen, 100.0)
    assert not watch.is_stable(p, seen, 105.0, stable=20)
    p.write_bytes(b"xx")  # grew
    assert not watch.is_stable(p, seen, 130.0, stable=20)
    assert watch.is_stable(p, seen, 151.0, stable=20)


def test_done_file_never_reprocessed(tmp_path, monkeypatch):
    folder = tmp_path / "in"
    folder.mkdir()
    _wav(folder / "a.wav", 1)
    monkeypatch.setattr(watch, "WATCH_DIR", folder)
    monkeypatch.setattr(watch, "STATE", tmp_path / "state.json")
    calls = []
    monkeypatch.setattr(watch, "process", lambda p, **k: calls.append(p) or p)
    seen, done = {}, set()
    watch.scan_once(seen, done)           # first sighting: not stable yet
    assert calls == []
    for v in list(seen):                  # pretend it has been quiet
        seen[v] = (seen[v][0], 0.0)
    watch.scan_once(seen, done)
    assert len(calls) == 1
    watch.scan_once(seen, done)
    assert len(calls) == 1                # done, not repeated
    assert watch.load_done(tmp_path / "state.json") == done
