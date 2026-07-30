"""Bump the version in pyproject and the plugin manifest, then tag the commit."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
PACKAGE = "deep-zotero"
EXTRA = "vision"

VERSION_LINE = re.compile(r'^(version = ")([^"]+)(")$', re.M)
PIN = re.compile(rf"^{re.escape(PACKAGE)}\[{re.escape(EXTRA)}\]==(.+)$")


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def read_versions() -> tuple[str, str, str]:
    """Return the versions in pyproject, the manifest, and the manifest's pin."""
    m = VERSION_LINE.search(PYPROJECT.read_text(encoding="utf-8"))
    if not m:
        sys.exit("no version line in pyproject.toml")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    args = manifest["mcpServers"][PACKAGE]["args"]
    pins = [PIN.match(a) for a in args]
    pin = next((p.group(1) for p in pins if p), None)
    if pin is None:
        sys.exit(f"no {PACKAGE}[{EXTRA}]==<version> pin in the manifest args")
    return m.group(2), manifest["version"], pin


def bump(version: str, part: str) -> str:
    try:
        major, minor, patch = (int(x) for x in version.split("."))
    except ValueError:
        sys.exit(f"cannot bump non-semver version {version!r}")
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def write_versions(new: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    text, count = VERSION_LINE.subn(rf"\g<1>{new}\g<3>", text, count=1)
    if count != 1:
        sys.exit("failed to rewrite the pyproject version")
    PYPROJECT.write_text(text, encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["version"] = new
    server = manifest["mcpServers"][PACKAGE]
    server["args"] = [
        f"{PACKAGE}[{EXTRA}]=={new}" if PIN.match(a) else a for a in server["args"]
    ]
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("version", nargs="?", help="explicit version, e.g. 0.3.0")
    target.add_argument("--bump", choices=("major", "minor", "patch"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-branch", action="store_true",
                   help="release from a branch other than main")
    args = p.parse_args()

    project, manifest, pin = read_versions()
    if len({project, manifest, pin}) != 1:
        sys.exit(f"versions already disagree: pyproject={project} "
                 f"manifest={manifest} pin={pin}")

    new = args.version or bump(project, args.bump)
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        sys.exit(f"{new!r} is not MAJOR.MINOR.PATCH")
    tag = f"v{new}"

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main" and not args.allow_branch:
        sys.exit(f"on {branch}, not main; pass --allow-branch to override")
    if git("tag", "--list", tag):
        sys.exit(f"{tag} already exists")

    print(f"{project} -> {new}  (tag {tag}, branch {branch})")
    if args.dry_run:
        print("dry run, nothing written")
        return

    # Only tracked changes matter; the release commit adds two known paths.
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("tracked files have uncommitted changes; commit them first")

    write_versions(new)
    git("add", str(PYPROJECT.relative_to(ROOT)), str(MANIFEST.relative_to(ROOT)))
    git("commit", "-m", f"chore: release {new}")
    git("tag", tag)
    print(f"committed and tagged {tag}")
    print(f"pushing the tag publishes to PyPI:\n  git push origin main {tag}")


if __name__ == "__main__":
    main()
