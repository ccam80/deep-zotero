"""Plugin manifest must not hand the server values it cannot resolve."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "plugin" / ".claude-plugin" / "plugin.json"
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
PLACEHOLDER = re.compile(r"\$\{[^}]+\}")


@pytest.fixture(scope="module")
def server() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return manifest["mcpServers"]["deep-zotero"]


@pytest.fixture(scope="module")
def shipped() -> Path:
    """The directory Claude Code copies into a user's plugin cache."""
    entry, = json.loads(MARKETPLACE.read_text(encoding="utf-8"))["plugins"]
    source = entry["source"]
    assert isinstance(source, str) and source.startswith("./"), (
        f"a relative-path source must start with './', got {source!r}"
    )
    return ROOT / source.removeprefix("./")


def test_no_unexpanded_placeholders_reach_the_server(server):
    """A ``${VAR}`` that the client cannot expand arrives as that literal string.

    Config treats any non-empty value as real, so a literal placeholder becomes a
    path or key instead of falling back to its default.
    """
    offenders = [
        f"{key}={value}"
        for key, value in (server.get("env") or {}).items()
        if PLACEHOLDER.search(str(value))
    ]
    offenders += [a for a in server["args"] if PLACEHOLDER.search(a)]
    assert not offenders, f"unexpanded placeholders: {offenders}"


def test_server_inherits_configuration_rather_than_restating_it(server):
    """Settings come from the environment Claude Code starts from."""
    assert "env" not in server


def test_command_is_uvx(server):
    assert server["command"] == "uvx"
    assert server["args"][:1] == ["--from"]


def test_marketplace_ships_a_subdirectory_rather_than_the_repository(shipped):
    assert shipped != ROOT
    assert shipped.is_dir()


def test_shipped_directory_holds_the_manifest_the_release_bumps(shipped):
    assert (shipped / ".claude-plugin" / "plugin.json") == MANIFEST
    assert MANIFEST.is_file()


def test_shipped_directory_carries_no_repository_payload(shipped):
    for directory in ("src", "tests", "tools", ".github", ".venv"):
        assert not (shipped / directory).exists(), f"{directory} would ship"
    strays = sorted(p.relative_to(shipped).as_posix()
                    for p in shipped.rglob("*.py"))
    assert not strays, f"python files would ship: {strays}"
