---
description: Set up deep-zotero after installing the plugin - environment variables, Tesseract, first index.
---

# Set up deep-zotero (agent runbook)

You are an AI coding agent setting up an installed deep-zotero plugin. Work through
the steps **in order** and verify each before moving on. The user should only have to
paste API keys and confirm system changes.

The plugin launches its MCP server through `uvx`, which fetches the pinned
`deep-zotero` wheel. There is nothing to clone, build, or register.

## 1. Preconditions

Confirm:
- `uv` is on `PATH` (`uv --version`). If missing, point the user at
  https://docs.astral.sh/uv/ and stop.
- The plugin is installed and its tools are visible. If not, `/plugin install
  deep-zotero` and restart Claude Code.
- Zotero 8 with PDFs in `storage/`. On Zotero 7 every citation key comes back empty.

Locate the Zotero data directory — it contains `zotero.sqlite` and `storage/`.
Default is `~/Zotero`. Confirm the path exists.

## 2. API keys

Ask the user for the keys they have:
- **Gemini** (embeddings, required by default) — https://aistudio.google.com/app/apikey
- **Anthropic** (vision table extraction) — https://console.anthropic.com/ . Without
  it, text and figures still index but **no tables are extracted**.

Do not echo the full keys back in your messages.

## 3. Environment variables

The server reads its whole configuration from the environment Claude Code starts
from. Set these four for the user's account, then have them restart Claude Code —
a newly set persistent variable only reaches new processes.

| Variable | Value |
|---|---|
| `DEEP_ZOTERO_DATA_DIR` | the Zotero data directory from step 1 |
| `DEEP_ZOTERO_CHROMA_PATH` | where the index should live, e.g. `~/.local/share/deep-zotero/chroma` |
| `GEMINI_API_KEY` | from step 2 |
| `ANTHROPIC_API_KEY` | from step 2, omit if the user opted out of vision |

Windows (PowerShell), per variable:

```powershell
[Environment]::SetEnvironmentVariable("DEEP_ZOTERO_DATA_DIR", "C:\Users\you\Zotero", "User")
```

macOS/Linux: add `export` lines to `~/.zshrc` or `~/.bashrc`.

To point at a config file instead, set `DEEP_ZOTERO_CONFIG` to its path. A config
file takes precedence over the environment wherever both supply a setting.

## 4. Tesseract-OCR (only if the user has scanned PDFs)

Text-based PDFs do **not** need this. With no scanned papers, skip to step 5 and
note that scanned pages are silently skipped.

1. Check whether Tesseract is installed:
   - Windows: look for `C:\Program Files\Tesseract-OCR\tesseract.exe` and
     `...\tessdata\eng.traineddata`.
   - macOS/Linux: `which tesseract`, then locate its `tessdata` directory (often
     `/usr/share/tesseract-ocr/*/tessdata` or `/opt/homebrew/share/tessdata`).
2. If missing, guide the user to install it — do not install system software
   without consent:
   - Windows: `winget install UB-Mannheim.TesseractOCR`, including English data.
   - macOS: `brew install tesseract`.
   - Debian/Ubuntu: `sudo apt install tesseract-ocr`.
3. PyMuPDF finds the language data through **`TESSDATA_PREFIX`**, pointing at the
   `tessdata` directory. This is a persistent system change, so **confirm with the
   user before setting it**, then set it alongside the step 3 variables.

Without `TESSDATA_PREFIX`, scanned pages report
`"OCR disabled because Tesseract language data not found."`

## 5. Index the library (first real run — spends money)

Nothing is searchable until this runs. It makes real Gemini and Anthropic calls, so
**confirm with the user first** — vision is roughly $0.016 per table through the
Haiku batch API.

Start small:

```bash
uvx --from "deep-zotero[vision]" deep-zotero-index --limit 5 -v
```

Confirm it reads the Zotero DB, extracts, embeds, and writes to ChromaDB without
errors. Then offer a full index by dropping `--limit`, or let the user trigger it
later through the `index_library` MCP tool.

## 6. Verify end to end

Call `get_index_stats` through the MCP server. It should report the documents and
chunks just indexed. Then call `search_papers` with a topic you saw in the indexed
titles and confirm a passage comes back with a citation key.

If the tools are missing, the server did not start: check `uv` is on `PATH` and that
the step 3 variables are visible to Claude Code's process.

## Done

Summarize for the user: which variables were set and where, Tesseract status, how
many documents were indexed, and that a fuller index can run later. Point them at
the README for the configuration and tool reference.
