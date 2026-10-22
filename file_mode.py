"""File mode: transcribe audio files with timestamps, in fixed audio chunks.

Decode (PyAV) and transcribe (the engine Walnut already picked in stt.py) work
on fixed-length chunks with an offset, through an iterator. Memory stays flat
for any file length, and a v2 live-audio source can feed the same
``transcribe_chunks`` by yielding ``(offset_seconds, float32_mono_16k_array)``.

Direct-read-vs-decode idea borrowed from Echo (https://github.com/alexsisk/echo,
MIT License, Copyright (c) 2026 Alex Sisk): formats libsndfile can read go
straight in, everything else (m4a, mp3, mp4) is decoded. Segment-timestamp
approach also from Echo app/transcribe.py: keep the engine's segments, not just
the joined text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SR = 16000
CHUNK_SECONDS = 120.0  # chosen from benchmark, see README "File mode"
DIRECT_FORMATS = {".wav", ".flac"}
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".flac", ".aac", ".caf", ".aiff", ".mov"}

Chunk = tuple[float, np.ndarray]


def iter_audio_chunks(path: str | Path, chunk_seconds: float = CHUNK_SECONDS,
                      sr: int = SR) -> Iterator[Chunk]:
    """Yield (offset_seconds, mono float32 samples at 16 kHz), chunk by chunk."""
    path = Path(path)
    size = int(chunk_seconds * sr)
    if path.suffix.lower() in DIRECT_FORMATS:
        import soundfile as sf
        with sf.SoundFile(str(path)) as f:
            if f.samplerate == sr:
                pos = 0
                while True:
                    block = f.read(size, dtype="float32", always_2d=True)
                    if len(block) == 0:
                        return
                    yield pos / sr, block.mean(axis=1)
                    pos += len(block)
        # other sample rates: fall through to the decoder, which resamples
    yield from _decode_chunks(path, size, sr)


def _decode_chunks(path: Path, size: int, sr: int) -> Iterator[Chunk]:
    import av

    pending: list[np.ndarray] = []
    have = 0
    pos = 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {path.name}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)

        def frames() -> Iterator[np.ndarray]:
            for packet in container.demux(stream):
                for frame in packet.decode():
                    for out in resampler.resample(frame):
                        yield out.to_ndarray().reshape(-1)
            for out in resampler.resample(None) or []:
                yield out.to_ndarray().reshape(-1)

        for arr in frames():
            pending.append(arr)
            have += len(arr)
            while have >= size:
                buf = np.concatenate(pending)
                yield pos / sr, buf[:size]
                pos += size
                rest = buf[size:]
                pending, have = ([rest] if len(rest) else []), len(rest)
    if have:
        yield pos / sr, np.concatenate(pending)


def _engine():
    """Return (run, model_id) using the backend and model stt.py would pick."""
    import stt
    backend = stt.resolve_backend("auto")
    model = stt.canonical("")
    entry = stt._BY_ID[model]
    lang = "en" if entry["english_only"] else None
    if backend == stt.MLX:
        import mlx_whisper
        repo = stt.model_name_for(model, backend)

        def run(audio: np.ndarray) -> list[dict]:
            r = mlx_whisper.transcribe(
                audio, path_or_hf_repo=repo, language=lang,
                condition_on_previous_text=False)
            return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in r.get("segments", [])]
    else:
        from faster_whisper import WhisperModel
        fw = WhisperModel(stt.model_name_for(model, backend), device="cpu",
                          compute_type="int8")

        def run(audio: np.ndarray) -> list[dict]:
            segs, _ = fw.transcribe(audio, language=lang,
                                    condition_on_previous_text=False, vad_filter=True)
            return [{"start": s.start, "end": s.end, "text": s.text.strip()}
                    for s in segs]
    return run, model


def transcribe_chunks(chunks: Iterable[Chunk], run=None) -> Iterator[dict]:
    """Yield segments {start, end, text} with absolute times. ``run`` is injectable."""
    if run is None:
        run, _ = _engine()
    for offset, audio in chunks:
        for seg in run(audio):
            if seg["text"]:
                yield {"start": seg["start"] + offset, "end": seg["end"] + offset,
                       "text": seg["text"]}


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def segments_to_lines(segments: Iterable[dict]) -> list[str]:
    return [f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments]


def slugify(name: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (s[:limit].strip("-")) or "audio"


def render_markdown(title: str, date: str, source: str, lines: list[str]) -> str:
    return f"Title: {title}\nDate: {date}\nSource: {source}\n====\n" + "\n".join(lines) + "\n"
