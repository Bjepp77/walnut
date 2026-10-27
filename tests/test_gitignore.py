"""Private data must never be one `git add -A` away from GitHub."""
import shutil
import subprocess

import pytest

from conftest import REPO


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(),
                    reason="needs the git checkout")
@pytest.mark.parametrize("name", [
    "walnut.db",
    "walnut.db-shm",
    "walnut.db-wal",
    "walnut.db.stale-intel-backup",
    "walnut.db.bak-20261027",
    "walnut.db.backup",
])
def test_database_and_every_backup_copy_is_ignored(name):
    # --no-index: judge the pattern alone, even for a path git still tracks
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", name], cwd=REPO)
    assert r.returncode == 0, f"{name} is not gitignored"
