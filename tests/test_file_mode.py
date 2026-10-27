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
    assert not (folder / "a.wav").exists() and (folder / "done" / "a.wav").exists()  # moved, not deleted
    watch.scan_once(seen, done)
    assert len(calls) == 1                # done, not repeated
    assert watch.load_done(tmp_path / "state.json") == done


def test_failing_file_moved_to_failed_after_3_attempts(tmp_path, monkeypatch):
    folder = tmp_path / "in"
    folder.mkdir()
    src = _wav(folder / "bad.wav", 1)
    monkeypatch.setattr(watch, "WATCH_DIR", folder)
    monkeypatch.setattr(watch, "STATE", tmp_path / "state.json")
    calls = []

    def boom(p, **k):
        calls.append(p)
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(watch, "process", boom)
    seen, done, fails = {}, set(), {}
    watch.scan_once(seen, done, fails)    # first sighting: not stable yet
    for v in list(seen):
        seen[v] = (seen[v][0], 0.0)
    for _ in range(2):
        watch.scan_once(seen, done, fails)
    assert len(calls) == 2 and src.exists()   # still retrying
    watch.scan_once(seen, done, fails)
    assert len(calls) == 3
    moved = folder / "failed" / "bad.wav"
    assert not src.exists() and moved.exists()  # moved, not deleted
    err = (folder / "failed" / "bad.wav.error.txt").read_text()
    assert "decoder exploded" in err and "3 attempts" in err
    watch.scan_once(seen, done, fails)
    assert len(calls) == 3                # failed/ is not rescanned
    assert done == set() and fails == {}


def test_quarantine_keeps_earlier_failure_of_same_name(tmp_path):
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed" / "x.m4a").write_text("old")
    p = tmp_path / "x.m4a"
    p.write_text("new")
    dest = watch.quarantine(p, "err")
    assert dest.name == "x-1.m4a" and dest.read_text() == "new"
    assert (tmp_path / "failed" / "x.m4a").read_text() == "old"
    assert (tmp_path / "failed" / "x-1.m4a.error.txt").read_text() == "err"


def test_dataless_icloud_file_is_fetched_not_failed(tmp_path, monkeypatch):
    folder = tmp_path / "in"
    folder.mkdir()
    src = _wav(folder / "memo.wav", 1)
    monkeypatch.setattr(watch, "WATCH_DIR", folder)
    monkeypatch.setattr(watch, "STATE", tmp_path / "state.json")
    fetched, calls = [], []
    monkeypatch.setattr(watch, "request_download", fetched.append)
    monkeypatch.setattr(watch, "process", lambda p, **k: calls.append(p) or p)
    dataless = {"on": True}
    monkeypatch.setattr(watch, "is_dataless", lambda p: dataless["on"])
    seen, done, fails = {}, set(), {}
    for _ in range(5):
        watch.scan_once(seen, done, fails)
    assert fetched == [src] * 5 and calls == [] and fails == {} and src.exists()
    dataless["on"] = False                # bytes landed
    watch.scan_once(seen, done, fails)    # first real sighting starts the timer
    for v in list(seen):
        seen[v] = (seen[v][0], 0.0)
    watch.scan_once(seen, done, fails)
    assert calls == [src]


# ------------------------------------------------- digest hand-off failures

def _watch_one(tmp_path, monkeypatch, digest_body):
    """One stable recording in a temp inbox, a fake transcriber, and a fake
    audio_digest.py whose body is digest_body (None means the script is missing)."""
    folder, out = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    src = _wav(folder / "memo.wav", 1)
    digest = tmp_path / "audio_digest.py"
    if digest_body is not None:
        digest.write_text(digest_body)
    monkeypatch.setattr(watch, "WATCH_DIR", folder)
    monkeypatch.setattr(watch, "OUT_DIR", out)
    monkeypatch.setattr(watch, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(watch, "DIGEST", digest)
    run = lambda audio: [{"start": 0.0, "end": 1.0, "text": "hello"}]
    seen, done, fails = {}, set(), {}
    watch.scan_once(seen, done, fails, run=run)       # first sighting
    for v in list(seen):
        seen[v] = (seen[v][0], 0.0)                    # pretend it went quiet
    made = watch.scan_once(seen, done, fails, run=run)
    return folder, src, made, done, fails


@pytest.mark.parametrize("digest_body", [
    "import sys; sys.stderr.write('boom'); sys.exit(3)\n",   # hand-off fails
    None,                                                     # script missing
])
def test_failed_handoff_leaves_recording_for_retry(tmp_path, monkeypatch, capsys,
                                                   digest_body):
    folder, src, made, done, fails = _watch_one(tmp_path, monkeypatch, digest_body)
    assert made == [] and done == set()                # not marked done
    assert src.exists() and not (folder / "done").exists()   # left for retry
    assert list(fails.values()) == [1]                 # counts toward failed/
    assert watch.load_done(tmp_path / "state.json") == set()
    err = capsys.readouterr().err
    assert "memo.wav" in err and "digest hand-off failed" in err   # reason logged
    if digest_body:
        assert "exit 3: boom" in err
    else:
        assert "not found" in err


def test_successful_handoff_marks_done_and_moves(tmp_path, monkeypatch):
    folder, src, made, done, fails = _watch_one(tmp_path, monkeypatch, "print('{}')\n")
    assert len(made) == 1 and len(done) == 1 and fails == {}
    assert not src.exists() and (folder / "done" / "memo.wav").exists()


def test_handoff_logs_name_and_status_not_the_transcript(tmp_path, monkeypatch, capsys):
    """audio_digest.py chunk prints the full transcript; that must not reach the log."""
    body = "print('{\"chunks\": [\"PRIVATE-TRANSCRIPT-TEXT\"]}')\n"
    folder, src, made, done, fails = _watch_one(tmp_path, monkeypatch, body)
    assert len(made) == 1
    logged = capsys.readouterr()
    assert "PRIVATE-TRANSCRIPT-TEXT" not in logged.out + logged.err
    assert f"{made[0].name}: handed off to audio-digest" in logged.out
