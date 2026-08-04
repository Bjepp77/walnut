"""Speech-to-text backends, chosen to fit the Mac Walnut is running on.

Walnut stores one *canonical* model id (``small.en``, ``large-v3-turbo``) in
walnut.db. This module translates that id into whatever the active engine
wants, so the same database and the same settings work on any Mac:

  Apple Silicon  ->  mlx-whisper, running on the GPU. Several times faster.
  Intel          ->  faster-whisper (CTranslate2), running int8 on the CPU.

Nothing else in Walnut needs to know which engine is in use. Call
``transcribe()`` and you get text.
"""

import importlib.util
import platform
import threading

import numpy as np

MLX = "mlx"
FASTER = "faster-whisper"

# Curated catalog. Every repo below was checked to exist for both engines;
# don't add an entry without confirming both sides resolve.
MODELS = [
    {"id": "large-v3-turbo", "label": "Large v3 Turbo", "size": "1.6 GB",
     "english_only": False, "mlx": "mlx-community/whisper-large-v3-turbo",
     "note": "Best accuracy. Fast on Apple Silicon, slow on Intel."},
    {"id": "medium.en", "label": "Medium (English)", "size": "1.5 GB",
     "english_only": True, "mlx": "mlx-community/whisper-medium.en-mlx",
     "note": "Very accurate English. Heavy on Intel."},
    {"id": "small.en", "label": "Small (English)", "size": "460 MB",
     "english_only": True, "mlx": "mlx-community/whisper-small.en-mlx",
     "note": "The sweet spot on Intel: fast and accurate enough."},
    {"id": "base.en", "label": "Base (English)", "size": "140 MB",
     "english_only": True, "mlx": "mlx-community/whisper-base.en-mlx",
     "note": "Fastest. Fumbles names and jargon."},
    {"id": "small", "label": "Small (multilingual)", "size": "460 MB",
     "english_only": False, "mlx": "mlx-community/whisper-small-mlx",
     "note": "Small, but understands ~99 languages."},
    {"id": "base", "label": "Base (multilingual)", "size": "140 MB",
     "english_only": False, "mlx": "mlx-community/whisper-base-mlx",
     "note": "Fastest multilingual option."},
]
_BY_ID = {m["id"]: m for m in MODELS}

# Walnut used to store raw MLX repo names. Keep old databases working.
_LEGACY = {
    "mlx-community/whisper-large-v3-turbo": "large-v3-turbo",
    "mlx-community/whisper-medium.en-mlx": "medium.en",
    "mlx-community/whisper-small.en-mlx": "small.en",
    "mlx-community/whisper-base.en-mlx": "base.en",
    "mlx-community/whisper-small-mlx": "small",
    "mlx-community/whisper-base-mlx": "base",
    "turbo": "large-v3-turbo",
    "large": "large-v3-turbo",
}

_cache: dict = {"key": None, "model": None}
_lock = threading.Lock()


# ------------------------------------------------------------------ hardware

def is_apple_silicon() -> bool:
    return platform.machine() == "arm64"


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def available_backends() -> list[str]:
    """Backends this machine can actually run, best first."""
    out = []
    if is_apple_silicon() and _installed("mlx_whisper"):
        out.append(MLX)
    if _installed("faster_whisper"):
        out.append(FASTER)
    return out


def resolve_backend(preference: str = "auto") -> str:
    """Turn a stored preference ('auto'/'mlx'/'faster-whisper') into a backend.

    An explicit choice that this Mac cannot honour falls back to what it can,
    rather than crashing — a database copied from another machine still works.
    """
    usable = available_backends()
    if not usable:
        raise RuntimeError(
            "No speech backend installed. Run `uv sync` in the Walnut folder.")
    if preference in usable:
        return preference
    return usable[0]


# -------------------------------------------------------------------- models

def canonical(model: str) -> str:
    """Map any stored/legacy model name onto a catalog id."""
    model = (model or "").strip()
    model = _LEGACY.get(model, model)
    return model if model in _BY_ID else default_model()


def default_model(backend: str | None = None) -> str:
    """Big model where it's cheap (GPU), small where it isn't (CPU)."""
    backend = backend or (MLX if is_apple_silicon() else FASTER)
    return "large-v3-turbo" if backend == MLX else "small.en"


def model_name_for(model: str, backend: str) -> str:
    """Catalog id -> the string the engine expects."""
    entry = _BY_ID[canonical(model)]
    return entry["mlx"] if backend == MLX else entry["id"]


def catalog(backend: str) -> list[dict]:
    """Model list for the UI, annotated for the active backend."""
    gpu = backend == MLX
    out = []
    for m in MODELS:
        out.append({
            "id": m["id"], "label": m["label"], "size": m["size"],
            "english_only": m["english_only"], "note": m["note"],
            "recommended": m["id"] == default_model(backend),
            # On CPU the big models are genuinely painful; flag them.
            "slow_here": not gpu and m["id"] in ("large-v3-turbo", "medium.en"),
        })
    return out


# ---------------------------------------------------------------- transcribe

def _load(model: str, backend: str):
    key = (backend, model)
    with _lock:
        if _cache["key"] != key:
            _cache["model"] = _MAKE[backend](model)
            _cache["key"] = key
        return _cache["model"]


def _make_mlx(model: str):
    import mlx_whisper

    repo = model_name_for(model, MLX)

    def run(audio, language, initial_prompt):
        # condition_on_previous_text feeds each chunk's output back in as
        # decoder context. On short dictation clips it buys nothing, and it's
        # what lets one stutter snowball into "...going to see if I'm going
        # to see if..." repeated forever. Off, a bad chunk stays contained.
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=repo,
            language=language or None, initial_prompt=initial_prompt,
            condition_on_previous_text=False)
        return result["text"].strip()

    return run


def _make_faster(model: str):
    # ctranslate2's OpenMP (libiomp5) segfaults intermittently on Intel Macs
    # when threads are unconstrained -- see SYSTRAN/faster-whisper#137. The
    # cap is an Intel workaround, so don't pay for it on Apple Silicon.
    import os

    intel = not is_apple_silicon()
    if intel:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from faster_whisper import WhisperModel

    engine = WhisperModel(
        model_name_for(model, FASTER), device="cpu", compute_type="int8",
        cpu_threads=4 if intel else 0)  # 0 = let ctranslate2 decide

    def run(audio, language, initial_prompt):
        # Same repetition-loop guard as the MLX path, plus VAD so trailing
        # silence isn't handed to the decoder (silence is the usual trigger
        # for hallucinated repeats).
        segments, _ = engine.transcribe(
            audio, language=language or None, initial_prompt=initial_prompt,
            condition_on_previous_text=False, vad_filter=True)
        return "".join(s.text for s in segments).strip()

    return run


_MAKE = {MLX: _make_mlx, FASTER: _make_faster}


def transcribe(audio: np.ndarray, model: str,
               language: str | None = None,
               initial_prompt: str | None = None,
               backend: str = "auto") -> str:
    backend = resolve_backend(backend)
    model = canonical(model)
    entry = _BY_ID[model]
    if entry["english_only"]:
        language = "en"
    return _load(model, backend)(audio, language, initial_prompt)


def describe(backend_pref: str = "auto") -> dict:
    """What the UI and `--doctor` show about this machine."""
    backend = resolve_backend(backend_pref)
    return {
        "machine": platform.machine(),
        "chip": "Apple Silicon" if is_apple_silicon() else "Intel",
        "backend": backend,
        "accelerator": "GPU (Metal)" if backend == MLX else "CPU (int8)",
        "available_backends": available_backends(),
        "default_model": default_model(backend),
    }
