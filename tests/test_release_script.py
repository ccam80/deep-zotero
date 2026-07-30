"""Drive tools/release.py against throwaway repositories."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "release.py"


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal repo holding the two files a release rewrites."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "release.py").write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "deep-zotero"\nversion = "0.2.0"\n', encoding="utf-8")
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text(json.dumps({
        "name": "deep-zotero",
        "version": "0.2.0",
        "mcpServers": {"deep-zotero": {"command": "uvx", "args": [
            "--from", "deep-zotero[vision]==0.2.0", "deep-zotero"]}},
    }, indent=2) + "\n", encoding="utf-8")
    for args in (("init", "-b", "main"), ("config", "user.email", "t@example.com"),
                 ("config", "user.name", "t"), ("add", "-A"), ("commit", "-m", "base")):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)
    return tmp_path


def release(repo: Path, *extra: str):
    return subprocess.run([sys.executable, "tools/release.py", "--bump", "patch", *extra],
                          cwd=repo, capture_output=True, text=True)


def test_untracked_file_does_not_block_a_release(repo):
    (repo / "session_handoff.md").write_text("notes\n", encoding="utf-8")
    assert release(repo).returncode == 0
    assert (repo / "session_handoff.md").read_text() == "notes\n"


def test_release_commit_touches_only_the_two_version_files(repo):
    (repo / "session_handoff.md").write_text("notes\n", encoding="utf-8")
    assert release(repo).returncode == 0
    touched = git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(touched) == [".claude-plugin/plugin.json", "pyproject.toml"]


def test_bump_rewrites_version_manifest_and_pin(repo):
    assert release(repo).returncode == 0
    manifest = json.loads((repo / ".claude-plugin/plugin.json").read_text())
    assert manifest["version"] == "0.2.1"
    assert "deep-zotero[vision]==0.2.1" in manifest["mcpServers"]["deep-zotero"]["args"]
    assert 'version = "0.2.1"' in (repo / "pyproject.toml").read_text()
    assert git(repo, "tag", "-l") == "v0.2.1"


def test_uncommitted_tracked_change_blocks_a_release(repo):
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "deep-zotero"\nversion = "0.2.0"\n# edited\n', encoding="utf-8")
    result = release(repo)
    assert result.returncode != 0
    assert "uncommitted" in (result.stdout + result.stderr)
    assert not git(repo, "tag", "-l")


def test_existing_tag_blocks_a_release(repo):
    subprocess.run(["git", "tag", "v0.2.1"], cwd=repo, capture_output=True, check=True)
    result = release(repo)
    assert result.returncode != 0
    assert "already exists" in (result.stdout + result.stderr)
