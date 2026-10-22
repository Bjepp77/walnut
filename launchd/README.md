# walnut-watch LaunchAgent

Runs `watch.py` at login and restarts it if it dies (60 s throttle).
Log: `~/Library/Logs/walnut-watch.log`.

Install:

    cp launchd/com.bjepp.walnut-watch.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.bjepp.walnut-watch.plist

Check: `launchctl print gui/$(id -u)/com.bjepp.walnut-watch | grep state`

Uninstall:

    launchctl bootout gui/$(id -u)/com.bjepp.walnut-watch
    mv ~/Library/LaunchAgents/com.bjepp.walnut-watch.plist ~/.Trash/

Paths are absolute (venv at ~/.local/venvs/walnut). Edit the plist if the repo moves.
