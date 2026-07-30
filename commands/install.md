---
description: Set up deep-zotero from a clone - venv, config, API keys, Tesseract, MCP registration.
---

# Install / set up deep-zotero (agent runbook)

You are an AI coding agent setting up this repository on the user's machine. Work
through the steps **in order**. After each step, verify it before moving on. The
user should only ever have to paste API keys and confirm system changes — **you**
do all file creation and JSON editing; never ask the user to hand-edit JSON.

Conventions used below:
- `PYEXE` = the venv interpreter. Windows: `.venv\Scripts\python.exe`. macOS/Linux:
  `.venv/bin/python`. Detect the OS and use the right one throughout.
- Run commands from the repo root.

---

## 0. Preconditions

1. Confirm Python 3.10+ is available: `python --version` (try `python3` if `python`
   is missing). If it's older than 3.10, stop and tell the user to install a newer
   Python.
2. Confirm you're in the repo root: `config.example.json` and `pyproject.toml`
   should both exist. If not, `cd` into the clone.

## 1. Create the virtual environment

```bash
python -m venv .venv
```

Verify `PYEXE` exists (`.venv\Scripts\python.exe` on Windows, `.venv/bin/python`
elsewhere).

## 2. Install the package (editable, with vision extras)

```bash
"PYEXE" -m pip install -e ".[vision]"
```

This pulls in the full pipeline plus the Anthropic/OpenAI vision deps. It may take a
few minutes. Verify with:

```bash
"PYEXE" -c "import deep_zotero; from deep_zotero import server; print('import OK')"
```

## 3. Tesseract-OCR (only needed for scanned / image-only PDFs)

Text-based PDFs do **not** need this — if the user has no scanned papers, you may
skip to step 4 and note that scanned pages will be silently skipped.

1. Check whether Tesseract is already installed:
   - Windows: look for `C:\Program Files\Tesseract-OCR\tesseract.exe` and
     `...\tessdata\eng.traineddata`.
   - macOS/Linux: `which tesseract` and locate its `tessdata` dir (often
     `/usr/share/tesseract-ocr/*/tessdata` or `/opt/homebrew/share/tessdata`).
2. If missing, guide the user to install it (do not install system software without
   consent):
   - Windows: `winget install UB-Mannheim.TesseractOCR` (or the UB-Mannheim
     installer). Include the English language data.
   - macOS: `brew install tesseract`.
   - Debian/Ubuntu: `sudo apt install tesseract-ocr`.
3. PyMuPDF finds the language data via the **`TESSDATA_PREFIX`** environment
   variable pointing at the `tessdata` directory. This is a persistent system
   change, so **confirm with the user before setting it**, then set it for their
   user account:
   - Windows (PowerShell):
     `[Environment]::SetEnvironmentVariable("TESSDATA_PREFIX", "C:\Program Files\Tesseract-OCR\tessdata", "User")`
   - macOS/Linux: add `export TESSDATA_PREFIX=/path/to/tessdata` to their shell
     profile (`~/.zshrc` / `~/.bashrc`).
   Note: a newly set persistent variable only reaches **new** processes — the user
   must restart their shell / MCP client for it to take effect.

Verify (in a shell that has the variable set):

```bash
TESSDATA_PREFIX="<tessdata path>" "PYEXE" -m pytest -q tests/test_ocr.py
```

It should pass. Without `TESSDATA_PREFIX`, that test fails with
`"OCR disabled because Tesseract language data not found."`

## 4. API keys

Ask the user for the keys they have. Both are optional but recommended:
- **Gemini** (embeddings) — https://aistudio.google.com/app/apikey . If the user
  doesn't want to use Gemini, set `embedding_provider` to `"local"` instead (lower
  quality, no key).
- **Anthropic** (vision table extraction) — https://console.anthropic.com/ . Without
  it, tables fall back to PyMuPDF heuristics.

Collect the key strings from the user; you will write them into `config.json` in the
next step. Do not echo the full keys back in your messages.

## 5. Create config.json

```bash
cp config.example.json config.json
```

Then **edit `config.json` yourself** (it's gitignored, so keys are never committed):
- Insert the user's `gemini_api_key` and `anthropic_api_key` (or set
  `"embedding_provider": "local"` / `"vision_enabled": false` if they opted out).
- Set `zotero_data_dir` to the user's Zotero data directory if it isn't the default
  `~/Zotero` (it contains `zotero.sqlite` and `storage/`). Confirm the path exists.
- Leave everything else at the example defaults unless the user asks otherwise.

Verify the config loads and validates:

```bash
"PYEXE" -c "from deep_zotero.config import Config; c=Config.load(); e=c.validate(); print('config path:', Config.default_config_path()); print('errors:', e or 'none')"
```

Resolve any validation errors before continuing (missing Zotero DB, missing required
key, etc.). Note: validation only checks a key is *present*, not that it's valid — a
real call in step 8 is the true test.

## 6. Register the MCP server

```bash
cp .mcp.json.example .mcp.json
```

Then **edit `.mcp.json` yourself**: set `mcpServers.deep-zotero.command` to the
**absolute** path of `PYEXE` (e.g. `C:\\...\\zotero_citation_mcp\\.venv\\Scripts\\python.exe`,
with escaped backslashes on Windows). `.mcp.json` is gitignored, so the
machine-specific path stays local.

Tell the user to **restart Claude Code / their MCP client** so it picks up the
server. Claude Code auto-loads a project-scoped `.mcp.json` from the repo root.

## 7. Verify the build

Run the suite (free — no API calls):

```bash
"PYEXE" -m pytest -q
```

Expected: the large majority pass. Known non-blocking failures even on a correct
setup are the vision/quality tests under `test_table_quality.py`,
`test_extraction_completeness.py`, `test_pdf_processor.py` (the `*_quality` ones),
and `test_extraction_integration.py` — their fixture extracts without a VisionAPI,
so they report 0 tables regardless of keys. Don't chase those here; step 8 is the
real end-to-end check. `test_ocr.py` passes only if `TESSDATA_PREFIX` is set in the
shell.

## 8. Index the library (first real run — spends tokens)

This is the true end-to-end validation: it makes real Gemini + Anthropic calls, so
**confirm with the user first** (vision is ~\$0.016/table via the Haiku batch API).
Start small:

```bash
"PYEXE" -m deep_zotero.cli --limit 5 -v
```

(The installed `deep-zotero-index` console script is equivalent if the venv is on
PATH.) Confirm it reads the Zotero DB, extracts, embeds, and writes to ChromaDB
without errors. Then offer a full index (`"PYEXE" -m deep_zotero.cli -v`) or let the
user trigger it later via the `index_library` MCP tool.

## Done

Summarize for the user: venv ready, package installed, Tesseract status, where
`config.json` and `.mcp.json` live (repo root, gitignored), and the reminder to
restart their MCP client. Point them at the README for the full configuration and
tool reference.
