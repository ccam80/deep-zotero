"""Plugin manifest must not hand the server values it cannot resolve."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
PLACEHOLDER = re.compile(r"\$\{[^}]+\}")


@pytest.fixture(scope="module")
def server() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return manifest["mcpServers"]["deep-zotero"]


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
