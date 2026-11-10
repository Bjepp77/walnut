"""Walnut core: narration, dictation, vocabulary, and global hotkeys.

All settings/vocabulary are read from the SQLite store at time of use, so
edits made in the web UI apply immediately — no restart needed (except for
hotkey changes, which reload the listener via HotkeyManager.reload()).
"""

import re
import subprocess
import threading
import time

import numpy as np

import permissions
import stt
import store

SAMPLE_RATE = 16_000

SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_STOP = "/System/Library/Sounds/Pop.aiff"
SOUND_ERROR = "/System/Library/Sounds/Basso.aiff"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def play_sound(path: str) -> None:
    subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def get_clipboard() -> str:
    return subprocess.run(["pbpaste"], capture_output=True, text=True).stdout


def set_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text, text=True)


def save_clipboard():
    """Snapshot the whole clipboard so it can be put back untouched.

    Through AppKit (PyObjC ships with rumps/pynput on macOS) every item and
    every type is kept, so a copied image, file or rich text survives. Without
    AppKit only plain text can be kept (pbpaste), which is the old behaviour.
    """
    try:
        from AppKit import NSPasteboard
    except ImportError:
        return ("text", get_clipboard())
    try:
        items = []
        for item in NSPasteboard.generalPasteboard().pasteboardItems() or []:
            kept = {}
            for t in item.types():
                data = item.dataForType_(t)
                if data is not None:
                    kept[str(t)] = data
            if kept:
                items.append(kept)
        return ("items", items)
    except Exception as e:           # never let a clipboard quirk kill a paste
        log(f"Could not snapshot the clipboard ({e}); keeping text only.")
        return ("text", get_clipboard())


def restore_clipboard(snap) -> None:
    kind, payload = snap
    if kind == "text":
        set_clipboard(payload)
        return
    from AppKit import NSPasteboard, NSPasteboardItem
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    new = []
    for kept in payload:
        item = NSPasteboardItem.alloc().init()
        for t, data in kept.items():
            item.setData_forType_(data, t)
        new.append(item)
    if new:
        pb.writeObjects_(new)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def transcribe(audio: np.ndarray, model: str,
               language: str | None = None,
               initial_prompt: str | None = None) -> str:
    """Transcribe with whichever engine suits this Mac (see stt.py)."""
    return stt.transcribe(audio, model, language=language,
                          initial_prompt=initial_prompt,
                          backend=store.get("stt_backend"))


class Vocabulary:
    """Live view over the store's dictionary/snippets."""

    @staticmethod
    def initial_prompt() -> str | None:
        hints = [w["word"] for w in store.words_list()]
        return ("Vocabulary: " + ", ".join(hints) + ".") if hints else None

    @staticmethod
    def apply(text: str) -> tuple[str, bool]:
        """Returns (processed_text, snippet_used)."""
        if store.get("snippets_enabled") == "1":
            norm = _normalize(text)
            for s in store.snippets_list():
                if s["enabled"] and norm == _normalize(s["trigger"]):
                    store.snippets_hit(s["id"])
                    return s["expansion"], True
        for r in store.replacements_list():
            # The replacement is a literal, not a template. Passing it straight
            # to re.sub() made "\1" or "\t" in a fix-up raise re.PatternError —
            # and because this runs in a worker thread, one bad rule silently
            # killed every transcription. A function replacement never expands.
            try:
                text = re.sub(re.escape(r["wrong"]), lambda _m, v=r["right"]: v,
                              text, flags=re.IGNORECASE)
            except re.error as e:                     # malformed `wrong` pattern
                log(f"skipping bad fix-up {r['wrong']!r}: {e}")
        return text, False


class Core:
    def __init__(self):
        from pynput import keyboard

        self.kb = keyboard.Controller()
        self.say_proc: subprocess.Popen | None = None
        self.play_proc: subprocess.Popen | None = None
        self.stream = None
        self.frames: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.on_state = lambda state: None  # 'idle' | 'recording' | 'busy'
        self.on_level = lambda rms: None    # mic level while recording
        # 'loading' until warm_up finishes. On a fresh machine that means a
        # ~1.6 GB download, during which the app otherwise looks inert.
        self.model_state = "loading"
        self.model_error = None
        self.on_command = lambda text: None  # console voice command transcript
        self.command_next = False  # route the next transcript to on_command

    # ------------------------------------------------------------ audio out
    # Two ways Walnut makes noise: `say` (narration) and `afplay` (replaying a
    # dictation recording). Both are held so the dashboard can stop them.

    def speaking(self) -> bool:
        return self.say_proc is not None and self.say_proc.poll() is None

    def playing(self) -> bool:
        return self.play_proc is not None and self.play_proc.poll() is None

    def busy_audio(self) -> bool:
        return self.speaking() or self.playing()

    def stop_speech(self) -> None:
        if self.speaking():
            self.say_proc.terminate()
        self.say_proc = None

    def stop_playback(self) -> None:
        if self.playing():
            self.play_proc.terminate()
        self.play_proc = None

    def stop_all(self) -> None:
        self.stop_speech()
        self.stop_playback()

    def play_file(self, path: str) -> None:
        """Replay a recorded dictation, keeping the handle so it can be cut."""
        self.stop_all()
        self.play_proc = subprocess.Popen(
            ["afplay", path], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

    def _copy_selection(self) -> str:
        from pynput.keyboard import Key

        saved = save_clipboard()
        try:
            set_clipboard("")
            time.sleep(0.30)  # let hotkey modifiers be released
            with self.kb.pressed(Key.cmd):
                self.kb.press("c")
                self.kb.release("c")
            time.sleep(0.30)
            return get_clipboard()
        finally:
            restore_clipboard(saved)   # restore even if the copy raises

    def speak(self, text: str, record_history: bool = True) -> None:
        self.stop_all()
        rate = int(store.get_valid("tts_rate"))
        voice = store.get_valid("tts_voice")
        cmd = ["say", "-r", str(rate)] + (["-v", voice] if voice else [])
        try:
            self.say_proc = subprocess.Popen(cmd + ["-f", "-"],
                                             stdin=subprocess.PIPE)
            self.say_proc.stdin.write(text.encode())
            self.say_proc.stdin.close()
        except (OSError, ValueError) as e:
            # a deleted voice, or `say` refusing the text
            log(f"Narration failed: {e}")
            play_sound(SOUND_ERROR)
            self.say_proc = None
            return
        if record_history:
            est = len(text.split()) / max(rate, 1) * 60
            store.history_add("readback", text, est)

    def toggle_speak(self) -> None:
        if self.speaking():
            self.stop_speech()
            log("Narration stopped.")
            return
        text = self._copy_selection().strip()
        if not text:
            log("No text selected.")
            play_sound(SOUND_ERROR)
            return
        log(f"Narrating {len(text.split())} words…")
        self.speak(text)

    # ------------------------------------------------------------ dictation

    def recording(self) -> bool:
        return self.stream is not None

    def warm_up(self) -> None:
        info = stt.describe(store.get("stt_backend"))
        model = stt.canonical(store.get("stt_model"))
        log(f"{info['chip']} detected → {info['backend']} on {info['accelerator']}")
        log(f"Loading speech model {model}… (first run downloads it, ~1.6 GB)")
        self.model_state = "loading"
        try:
            transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), model)
        except Exception as e:
            self.model_state = "error"
            self.model_error = f"{type(e).__name__}: {e}"
            log(f"Speech model failed to load: {self.model_error}")
            raise
        self.model_state = "ready"
        log("Speech model ready.")

    def toggle_dictate(self) -> None:
        # Non-blocking: transcription can take seconds while holding this lock.
        # Blocking here meant a hotkey press during transcription queued up and
        # then started an unexpected recording once the lock freed.
        if not self.lock.acquire(blocking=False):
            log("Still working on the last dictation — ignoring.")
            return
        try:
            if self.recording():
                self._stop_and_type()
            else:
                self._start_recording()
        finally:
            self.lock.release()

    def toggle_command(self) -> None:
        """Like dictation, but the transcript goes to the Console bridge
        (on_command) instead of being typed into the frontmost app."""
        if not self.lock.acquire(blocking=False):
            log("Still working on the last dictation — ignoring.")
            return
        try:
            if self.recording():
                self._stop_and_type()
            else:
                self.command_next = True
                try:
                    self._start_recording()
                finally:
                    # No recording means _stop_and_type will never reset the
                    # flag, and the next plain dictation would go to the Console.
                    if not self.recording():
                        self.command_next = False
        finally:
            self.lock.release()

    def _start_recording(self) -> None:
        import sounddevice as sd

        self.stop_all()
        self.frames = []

        def on_audio(indata, *_):
            self.frames.append(indata.copy())
            try:
                self.on_level(float(np.sqrt(np.mean(indata ** 2))))
            except Exception:
                pass

        def open_stream():
            s = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                               dtype="float32", callback=on_audio)
            try:
                s.start()
            except Exception:
                s.close()          # never leave a half-open stream behind
                raise
            return s

        # If opening the mic fails (permission denied, device in use), leave
        # self.stream as None. Assigning first meant recording() reported True
        # forever and the next hotkey press tried to stop a dead stream.
        try:
            stream = open_stream()
        except Exception as first:
            # PortAudio enumerates audio devices once, when it initialises. If
            # the default input changed since Walnut started — headphones in, a
            # Bluetooth mic, a meeting grabbing the device — those cached
            # indices go stale and EVERY open fails with a bare internal error.
            # The process is then dead to dictation until it restarts, which is
            # exactly what it looked like: hotkeys firing, nothing recording.
            # Rescanning is cheap; do it and try once more.
            log(f"Microphone open failed ({first}); rescanning audio devices…")
            try:
                sd._terminate()
                sd._initialize()
                stream = open_stream()
            except Exception as second:
                log(f"Could not open the microphone: {second}")
                play_sound(SOUND_ERROR)
                self.on_state("idle")
                return
            log("Recovered after rescanning audio devices.")
        self.stream = stream
        play_sound(SOUND_START)
        self.on_state("recording")
        log("Recording…")

    def _stop_and_type(self) -> None:
        was_command = self.command_next
        self.command_next = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            log(f"Error closing the microphone: {e}")
        self.stream = None
        play_sound(SOUND_STOP)
        self.on_state("busy")
        try:
            if not self.frames:
                log("No audio captured.")
                return
            audio = np.concatenate(self.frames)[:, 0]
            secs = len(audio) / SAMPLE_RATE
            if secs < 0.4:
                log("Recording too short, ignored.")
                return
            log(f"Transcribing {secs:.1f}s…")
            t0 = time.time()
            try:
                text = transcribe(
                    audio,
                    store.get("stt_model"),
                    language=store.get("stt_language") or None,
                    initial_prompt=Vocabulary.initial_prompt(),
                )
            except Exception as e:
                # model download failed, disk full, corrupt weights… don't die
                # silently in a worker thread: chime and say so in the log.
                log(f"Transcription failed: {type(e).__name__}: {e}")
                play_sound(SOUND_ERROR)
                return
            text, used_snippet = Vocabulary.apply(text)
            log(f"({time.time() - t0:.1f}s) → {text!r}")
            if not text:
                return
            audio_path = self._save_audio(audio)
            if was_command:
                store.history_add("command", text, secs,
                                  audio_path=audio_path)
                threading.Thread(target=self.on_command, args=(text,),
                                 daemon=True).start()
                return
            store.history_add("dictation", text, secs,
                              snippet=used_snippet, audio_path=audio_path)
            store.prune_recordings()   # keep only the newest few .wav files
            self._type(text + " ")
        finally:
            self.on_state("idle")

    @staticmethod
    def _save_audio(audio: np.ndarray) -> str:
        import soundfile as sf

        # Second resolution collided: two dictations inside the same second
        # wrote the same file, so one clip overwrote the other and deleting
        # either history row unlinked the survivor's audio too.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = store.RECORDINGS / f"dictation-{stamp}.wav"
        n = 1
        while path.exists():
            path = store.RECORDINGS / f"dictation-{stamp}-{n}.wav"
            n += 1
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        return str(path)

    def _type(self, text: str) -> None:
        from pynput.keyboard import Key

        saved = save_clipboard()
        try:
            set_clipboard(text)
            time.sleep(0.25)   # slower apps read the clipboard late; 0.15 dropped pastes
            with self.kb.pressed(Key.cmd):
                self.kb.press("v")
                self.kb.release("v")
            time.sleep(0.75)   # let the paste land before the clipboard is restored
        finally:
            restore_clipboard(saved)   # never strand the transcript in the clipboard


class HotkeyManager:
    """Owns the pynput GlobalHotKeys listener; can reload after key changes."""

    def __init__(self, core: Core):
        self.core = core
        self.listener = None

    def start(self) -> None:
        """Bind the hotkeys. Never raises: an unusable combo falls back to the
        default rather than killing the listener (or, at launch, the process —
        which launchd would then respawn forever)."""
        from pynput import keyboard

        self.stop()
        # get_valid() heals a database holding a combo pynput cannot parse
        speak = store.get_valid("hotkey_speak")
        dictate = store.get_valid("hotkey_dictate")
        command = store.get_valid("hotkey_command")

        def on_speak():
            threading.Thread(target=self.core.toggle_speak, daemon=True).start()

        def on_dictate():
            threading.Thread(target=self.core.toggle_dictate, daemon=True).start()

        def on_command():
            threading.Thread(target=self.core.toggle_command, daemon=True).start()

        try:
            self.listener = keyboard.GlobalHotKeys({speak: on_speak,
                                                    dictate: on_dictate,
                                                    command: on_command})
            self.listener.start()
            # pynput binds happily without Accessibility and then never fires.
            # Saying "Hotkeys active" here would be a lie, and it was the single
            # most common way Walnut appeared broken to a new user.
            if not permissions.accessibility_trusted():
                log(f"Hotkeys registered ({speak}, {dictate}, {command}) but "
                    f"they will NOT fire: Walnut has no Accessibility permission.")
                log("Fix: System Settings → Privacy & Security → Accessibility, "
                    "add Walnut (or your terminal), then restart Walnut.")
                return
        except Exception as e:
            # e.g. the combos collide, or macOS refuses the tap
            self.listener = None
            log(f"Could not bind hotkeys ({speak}, {dictate}, {command}): {e}")
            if (speak, dictate, command) == (store.DEFAULTS["hotkey_speak"],
                                             store.DEFAULTS["hotkey_dictate"],
                                             store.DEFAULTS["hotkey_command"]):
                log("Defaults failed too — check Accessibility permission.")
                return
            log("Falling back to the default hotkeys.")
            store.set_setting("hotkey_speak", store.DEFAULTS["hotkey_speak"])
            store.set_setting("hotkey_dictate", store.DEFAULTS["hotkey_dictate"])
            store.set_setting("hotkey_command", store.DEFAULTS["hotkey_command"])
            self.start()
            return
        log(f"Hotkeys active: {speak} (narrate), {dictate} (dictate), "
            f"{command} (console command)")

    def stop(self) -> None:
        if self.listener:
            self.listener.stop()
            self.listener = None

    def reload(self) -> None:
        self.start()
