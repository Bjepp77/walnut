<div align="center">

<img src="static/apple-touch-icon.png" width="96" alt="Walnut">

# Walnut

**Talk to your Mac. It types. Select anything. It reads.**

Local voice dictation and narration for macOS — the whole loop, on your own
machine. No cloud, no API keys, no account, no subscription.

`⌃⌥D` to dictate into any app · `⌃⌥S` to hear any selection read aloud

<sub>MIT licensed · Apple Silicon & Intel · Python 3.12+ · ~1.6 GB model, downloaded once</sub>

</div>

<br>

<div align="center">
  <img src="docs/dashboard.png" width="880" alt="The Walnut dashboard">
</div>

<br>

---

## Why

Commercial dictation tools are excellent and they all ship your voice somewhere
else. Walnut does the same job with Whisper running on your own silicon. Your
audio never leaves the machine, there is nothing to sign up for, and the whole
thing is about 1,300 lines of Python you can read in an afternoon.

It also reads back. Select a paragraph in any app, press `⌃⌥S`, and macOS's own
speech engine narrates it. Two hotkeys, one loop: **your words in, your words
out.**

## Install

```sh
git clone https://github.com/Bjepp77/walnut.git
cd walnut
./install.sh --app --login
```

That installs [uv](https://docs.astral.sh/uv/) if you need it, resolves
dependencies, runs a self-test, builds `Walnut.app`, and starts Walnut at login.
Every flag is optional — bare `./install.sh` just sets up the code.

| Flag | Adds |
|:--|:--|
| *(none)* | dependencies + self-test. Run it with `uv run walnut.py`. |
| `--app` | `Walnut.app` in /Applications — double-click, Spotlight, Launchpad. |
| `--login` | starts Walnut automatically at login. |
| `--uninstall` | removes the app and the login item. Your data stays. |

A squirrel appears in the menu bar. The dashboard lives at
**http://127.0.0.1:8765**.

> **First run downloads the speech model** — about 1.6 GB on Apple Silicon,
> 460 MB on Intel. Once, ever. The dashboard shows a banner while it works.

### Grant one permission

macOS requires **Accessibility** for global hotkeys and for typing into other
apps. Walnut cannot grant it for you.

> System Settings → Privacy & Security → Accessibility → add **Walnut** → restart Walnut

Without it the hotkeys register and silently never fire — so Walnut checks, and
tells you, in the menu bar and on the dashboard. It will not pretend to work.

The microphone prompt appears the first time you dictate. Click Allow.

## It fits itself to your Mac

You configure nothing. Walnut detects the chip, picks the engine, and picks a
model that is genuinely pleasant on that hardware.

| Your Mac | Engine | Runs on | Default model | Status |
|:--|:--|:--|:--|:--|
| Apple Silicon (M1–M5) | `mlx-whisper` | GPU (Metal) | `large-v3-turbo` | tested |
| Intel | `faster-whisper` | CPU, int8 | `small.en` | **implemented, unverified** |

Walnut stores one canonical model name and translates it per engine, so the same
`walnut.db` works if you carry it between an Intel and an M-series Mac.

> **Intel Macs:** the code path exists and its dependencies resolve, but Walnut
> has never actually been run on Intel silicon. If that's you, you're the first
> — please open an issue with whatever breaks. Everything else here is tested.

## The hotkeys

| Hotkey | What happens |
|:--|:--|
| `⌃⌥D` | Chime. Speak. Chime. Your words are typed into the frontmost app. |
| `⌃⌥S` | Whatever text you have selected, anywhere, is read aloud. Press again to stop. |
| `⌃⌥C` | Chime. Speak. Chime. The phrase goes to the MegaMind Console instead of being typed (see Voice commands below). |

All are re-bindable on the **Shortcuts** page — live, no restart. Dictation is
`⌃⌥D` rather than `⌃⌥Space` because macOS uses that to switch input sources.

## Make the narration sound human

**Do this. It takes two minutes and it is the single biggest improvement
available.**

Every Mac defaults to a formant synthesiser from the mid-2000s — the robot you
are picturing. macOS also ships modern neural voices, free and fully offline,
but does not install them and buries the download three levels deep. Most people
never find out they exist.

> System Settings → Accessibility → **Read & Speak** → System voice → Manage Voices
>
> *(on macOS 15 and earlier the pane is called **Spoken Content**)*

Download anything marked **(Premium)** or **(Enhanced)** — Ava, Zoe, Evan, Lee.
Then pick it in Walnut's **Settings**, where they're grouped at the top under
*High quality — neural*.

Walnut checks for this on every launch. If you have no good voice installed it
says so and offers to open the pane. If you have one but haven't selected it, it
names it. If the voice you picked is later uninstalled — `say` substitutes the
default *silently* — Walnut notices that too. It will not pretend to sound good.

Siri's voices are walled off from the `say` command entirely and cannot be used
by any app; Premium is the ceiling. Walnut deliberately ships no bundled neural
TTS: every open phonemiser in reach (`espeak-ng`, `phonemizer-fork`, and so
`piper-tts` and `kokoro-onnx`) is GPL-3, which would silently relicense this
MIT project.

## Voice commands (MegaMind Console)

Walnut can trigger headless skill runs on the MegaMind Console
(`Claude_Cowork/Projects/console`, must be running — `npm start` there, or
its launchd agent). Trigger + confirm only; no conversations.

End-to-end demo: press `⌃⌥C`, say **"vet this https://github.com/x/y"**,
press `⌃⌥C` again. Walnut sends the raw phrase to the Console's `/api/run`,
which matches it against the command-center trigger phrases and starts the
run. Walnut says "Running vet skill.", follows the run's stream, and when it
finishes speaks the one-line TL;DR of the result.

- **Ambiguous phrase** (matches two triggers, e.g. "email brief"): Walnut
  speaks the top two candidates and asks you to press the hotkey and say
  which one. It never guesses.
- **No match**: Walnut says "This needs a real session" — open a terminal or
  Cowork for anything conversational.
- **Console not running**: Walnut tells you so.

Settings live in the store like everything else: `hotkey_command` (default
`<ctrl>+<alt>+c`) and `console_url` (default `http://127.0.0.1:4173`), both
changeable via `PUT /api/settings` on the dashboard API.

## The dashboard

<table>
<tr><td width="150"><b>Dashboard</b></td><td>Streak, speaking vs typing WPM, time saved, words dictated, keystrokes you didn't type, activity charts.</td></tr>
<tr><td><b>Dictionary</b></td><td>Words Whisper should know — names, acronyms, jargon — plus fix-ups like <code>"hub spot"</code> → <code>HubSpot</code>. Hints are fed to the model, which rescues proper nouns on the smaller ones.</td></tr>
<tr><td><b>Snippets</b></td><td>Say a phrase, Walnut types the expansion. Your email, your sign-off, that URL you can never remember.</td></tr>
<tr><td><b>History</b></td><td>Every session: search, copy, replay the original audio, reveal in Finder, delete.</td></tr>
<tr><td><b>Settings</b></td><td>Voice, speed, language, typing WPM, engine, model — annotated with what your Mac chose and which models will be slow on it.</td></tr>
</table>

Walnut ships blank. No sample vocabulary, no snippets — make it yours.

## Choosing a model

Sizes are the download. All of them run entirely on your machine.

| Model | Size | Notes |
|:--|:--|:--|
| `large-v3-turbo` | 1.6 GB | Best accuracy, multilingual. Superb on Apple Silicon, slow on Intel. |
| `medium.en` | 1.5 GB | Very accurate English. Heavy on Intel. |
| `small.en` | 460 MB | The sweet spot on Intel. |
| `base.en` | 140 MB | Fastest. Fumbles names — lean on the Dictionary. |
| `small` / `base` | 460 / 140 MB | Multilingual equivalents. |

Models are cached by Hugging Face under `~/.cache/huggingface`. You can pin the
engine on the Settings page if you want to force `faster-whisper` on Apple
Silicon — useful for comparing.

## Privacy

Everything is local. There is no server, no telemetry, and no account.

- Transcripts and settings live in `walnut.db` (SQLite), next to the code.
- Dictation audio lives in `recordings/` — capped at the **newest 3 clips** by
  default, configurable, `0` to keep none.
- The dashboard binds `127.0.0.1` only, and refuses any request whose `Host`
  isn't loopback, so a web page you visit can't reach it.
- The dashboard has no login. Any program running on this Mac can read your
  dictation history through it, delete it, or change settings. That is fine
  for a Mac only you use; do not run Walnut on a shared machine.
- The single network call Walnut ever makes is downloading the Whisper model
  from Hugging Face, once.

## When something breaks

```sh
uv run walnut.py --doctor    # chip, engine, model, permissions, paths, log
uv run walnut.py --test      # end-to-end check, no mic needed
uv run walnut.py --version
```

`--doctor` is the first thing to run and the first thing to paste into an issue.
It exits non-zero if Accessibility is missing.

Hotkeys doing nothing is almost always Accessibility — granted to the wrong
binary, or not at all. Whatever launches Walnut is what needs the permission:
your terminal if you run it by hand, `uv` at login, `Walnut.app` if you
double-click it. Granting all three is fine.

Logs go to `/tmp/walnut.log` when Walnut starts at login.

## Development

```sh
uv run --group dev pytest -q     # 83 tests
```

Every test in `tests/` is a bug Walnut actually shipped: a dictionary fix-up
containing a backslash that silently killed all dictation, a hotkey that
crash-looped the app, leaked SQLite handles, a prune that deleted the recording
it was in the middle of saving. If one fails, it is telling you the truth.

`tests/test_first_run.py` forces the state a stranger's Mac is in — permission
denied, empty database, model still downloading. Two crash-loop bugs shipped
because nothing ever did that.

```
walnut.py    entry point, --test, --doctor    core.py         dictation, narration, hotkeys
stt.py       engine + model selection         store.py        SQLite: settings, history, stats
permissions.py  macOS Accessibility           server.py       Flask API
app.py       menu bar app (rumps)             overlay.py      the floating recording pill
install.sh   installer                        static/index.html  the dashboard
console_bridge.py   voice commands → MegaMind Console
```

`config.toml` is read once, on first run, to seed the database. After that the
dashboard is the source of truth. Delete `walnut.db` to start over.

## Requirements

macOS — Walnut uses the built-in `say` and `afplay`. Python 3.12 or 3.13, which
`uv` handles for you.

## License

[MIT](LICENSE). Take it, change it, ship it.

<div align="center"><sub>❦</sub></div>
