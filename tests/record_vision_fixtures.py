"""Record live vision responses into tests/fixtures/vision_responses.json.

Table extraction is vision-only, so offline tests need a recorded transcript to
produce any tables at all. conftest replays this file through the real parser.

Re-record after any change to caption detection, crop geometry, the system
prompt, or the model; a stale fixture raises a KeyError naming the table_id.

    .venv/Scripts/python.exe tests/record_vision_fixtures.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
_PAPERS_DIR = _TESTS_DIR / "fixtures" / "papers"
_OUT_PATH = _TESTS_DIR / "fixtures" / "vision_responses.json"
_PAPER_NAMES = ["noname1.pdf", "noname2.pdf", "noname3.pdf"]


class _RecordingVisionAPI:
    """Wraps a real VisionAPI, capturing every raw response by table_id."""

    def __init__(self, inner):
        self._inner = inner
        self.recorded: dict[str, str] = {}

    def extract_tables_batch(self, specs):
        responses = self._inner.extract_tables_batch(specs)
        for spec, resp in zip(specs, responses):
            self.recorded[spec.table_id] = resp.raw_response
        return responses


def _resolve_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    config_path = Path(
        os.environ.get("DEEP_ZOTERO_CONFIG", _TESTS_DIR.parent / "config.json")
    )
    if config_path.exists():
        key = json.loads(config_path.read_text(encoding="utf-8")).get("anthropic_api_key")
        if key:
            return key
    print(
        "ERROR: no Anthropic API key. Set ANTHROPIC_API_KEY or put "
        "anthropic_api_key in config.json.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from deep_zotero.feature_extraction.vision_api import VisionAPI
    from deep_zotero.pdf_processor import extract_document, resolve_pending_vision

    api = _RecordingVisionAPI(
        VisionAPI(
            api_key=_resolve_api_key(),
            cost_log_path=_TESTS_DIR / "_record_vision_costs.json",
        )
    )

    # Stems, not filenames: the doc key lands in the Batch API custom_id, which
    # must match ^[a-zA-Z0-9_-]{1,64}$ — a dot 400s the request.
    extractions = {}
    for name in _PAPER_NAMES:
        pdf = _PAPERS_DIR / name
        if not pdf.exists():
            print(f"ERROR: missing fixture PDF {pdf}", file=sys.stderr)
            return 1
        print(f"Extracting {name}...")
        extractions[pdf.stem] = extract_document(pdf, vision_api=api)

    print("\nSubmitting vision batches (this can take 10-30 minutes)...")
    resolve_pending_vision(extractions, vision_api=api)

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(
        json.dumps(api.recorded, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\nRecorded {len(api.recorded)} responses to {_OUT_PATH}")
    for stem, ext in extractions.items():
        print(f"  {stem}: {len(ext.tables)} tables, grade {ext.completeness.grade}")
    print(f"  session cost: ${api._inner.session_cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
