# DeepZotero

Semantic search over a Zotero library. PDFs are extracted (text, tables, figures), chunked, embedded, and stored in ChromaDB. An MCP server exposes the index to Claude Code (or any MCP client) as 13 tools for semantic search, boolean search, table/figure search, context expansion, citation graph lookup, indexing, and cost tracking.

## What it extracts

- **Text** — section-aware chunks with overlap, classified by document section (abstract, methods, results, etc.)
- **Tables** — vision-based extraction via Claude Haiku 4.5. Each table is rendered to PNG and transcribed to structured markdown (headers, rows, footnotes). Table extraction is vision-only: with vision disabled, or without an Anthropic key, no tables are extracted.
- **Figures** — detected with captions, extracted as PNGs, searchable by caption text.

## Requirements

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/app/apikey) for embeddings (unless using `embedding_provider: "local"`)
- An [Anthropic API key](https://console.anthropic.com/) for vision-based table extraction (optional but recommended)
- A Zotero installation with PDFs in `storage/`
- **Tesseract-OCR** — only needed to OCR scanned / image-only PDF pages. Install [Tesseract](https://github.com/tesseract-ocr/tesseract) with the language data you need, then set the `TESSDATA_PREFIX` environment variable to its `tessdata` directory (e.g. `C:\Program Files\Tesseract-OCR\tessdata`). PyMuPDF locates the OCR data via that variable; without it, scanned pages are skipped (`"OCR disabled because Tesseract language data not found."`). Text-based PDFs do not need Tesseract.

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[vision]"
```

The `vision` extra pulls in the Anthropic and OpenAI clients. Without it (`pip install -e .`) the pipeline still indexes text and figures, but no tables.

`commands/install.md` is a step-by-step setup runbook written for a coding agent — point Claude Code at it to have the venv, config, Tesseract, and MCP registration set up for you.

## Setup

### 1. Configuration

```bash
cp config.example.json config.json
```

Edit `config.json` (lives in the repo root, next to `config.example.json`; gitignored so your keys are never committed):

```json
{
    "zotero_data_dir": "~/Zotero",
    "chroma_db_path": "~/.local/share/deep-zotero/chroma",
    "gemini_api_key": "YOUR_GEMINI_KEY",
    "anthropic_api_key": "YOUR_ANTHROPIC_KEY"
}
```

All other fields have sensible defaults. You can also set `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` as environment variables instead. To load the config from a different location, point the `DEEP_ZOTERO_CONFIG` environment variable at it, or pass `--config PATH` to the CLI.

`config.json` is optional. Installed from a wheel rather than a clone there is no repo root to hold one, so the two path settings also read `DEEP_ZOTERO_DATA_DIR` and `DEEP_ZOTERO_CHROMA_PATH` from the environment. That lets a launcher supply every setting without writing a config file. A `config.json` still takes precedence over the environment wherever both are present.

### 2. API keys

**Gemini (required for default embeddings):**
Get a key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). Set it as `gemini_api_key` in config or `GEMINI_API_KEY` env var. If you don't want to use Gemini, set `"embedding_provider": "local"` to use ChromaDB's built-in all-MiniLM-L6-v2 model (no API key needed, lower quality).

**Anthropic (required for vision table extraction):**
Get a key at [console.anthropic.com](https://console.anthropic.com/). Set it as `anthropic_api_key` in config or `ANTHROPIC_API_KEY` env var. Table extraction is vision-only — without this key, text and figures are still indexed but **no tables are extracted**. Vision extraction uses the Anthropic Batch API with Claude Haiku 4.5 — cost is roughly $0.016 per table, with prompt caching reducing cost on large batches.

To disable vision extraction entirely:

```json
{
    "vision_enabled": false
}
```

### 3. Index your library

```bash
deep-zotero-index -v
```

To test with a subset first:

```bash
deep-zotero-index --limit 10 -v
```

This reads the Zotero SQLite database (read-only, safe while Zotero is open), extracts text/tables/figures from each PDF, chunks the text, embeds via Gemini, and stores everything in ChromaDB.

CLI options:

| Flag | Description |
|------|-------------|
| `--force` | Delete and rebuild index for all matching items |
| `--limit N` | Only index N items |
| `--item-key KEY` | Index a single Zotero item |
| `--title PATTERN` | Regex filter on title (case-insensitive) |
| `--no-vision` | Skip vision table extraction for this run |
| `--config PATH` | Use a different config file |
| `-v` | Debug logging |

The indexer is incremental — it only processes items not already in the index. Use `--force` after changing `chunk_size`, `embedding_dimensions`, or `ocr_language`.

You can also trigger indexing from the MCP client via the `index_library` tool.

### 4. Register the MCP server

The repo ships `.mcp.json.example` as a template (the real `.mcp.json` is gitignored because the interpreter path is machine-specific). Copy it and set the `command` to your venv's Python:

```bash
cp .mcp.json.example .mcp.json
```

```json
{
    "mcpServers": {
        "deep-zotero": {
            "command": "C:\\path\\to\\zotero_citation_mcp\\.venv\\Scripts\\python.exe",
            "args": ["-m", "deep_zotero.server"]
        }
    }
}
```

On macOS/Linux the interpreter is `/path/to/zotero_citation_mcp/.venv/bin/python`. Claude Code auto-loads a project-scoped `.mcp.json` from the repo root; alternatively, put the same `mcpServers` block in `~/.claude/settings.json`.

If you need scanned-page OCR, make sure `TESSDATA_PREFIX` (see [Requirements](#requirements)) is set in the environment the server runs in.

Restart Claude Code. All 13 tools will be available.

---

## Configuration reference

### Zotero

| Field | Default | Description |
|---|---|---|
| `zotero_data_dir` | `~/Zotero` | Path to Zotero's data directory (contains `zotero.sqlite` and `storage/`). Falls back to `DEEP_ZOTERO_DATA_DIR` env var |
| `chroma_db_path` | `~/.local/share/deep-zotero/chroma` | Where the ChromaDB index is stored on disk. Falls back to `DEEP_ZOTERO_CHROMA_PATH` env var |

### Embedding

| Field | Default | Description |
|---|---|---|
| `embedding_provider` | `"gemini"` | `"gemini"` for Gemini API, `"local"` for ChromaDB's built-in all-MiniLM-L6-v2 (no key needed) |
| `embedding_model` | `"gemini-embedding-001"` | Gemini model name (only used when provider is `"gemini"`) |
| `embedding_dimensions` | `768` | Output vector dimensions. `gemini-embedding-001` supports 64-3072. Changing requires `--force` re-index |
| `gemini_api_key` | `null` | Falls back to `GEMINI_API_KEY` env var |
| `embedding_timeout` | `120.0` | Timeout in seconds for embedding API calls |
| `embedding_max_retries` | `3` | Max retries for failed embedding calls |
| `embedding_rate_limit_backoff` | `30.0` | Seconds to wait before retrying after an HTTP 429 (per-minute quota) |

### Chunking

| Field | Default | Description |
|---|---|---|
| `chunk_size` | `400` | Target chunk size in tokens (~4 chars/token). Changing requires `--force` re-index |
| `chunk_overlap` | `100` | Overlap between consecutive chunks in tokens |

### Vision

| Field | Default | Description |
|---|---|---|
| `vision_enabled` | `true` | Enable vision table extraction during indexing |
| `vision_model` | `"claude-haiku-4-5-20251001"` | Anthropic model for table transcription |
| `anthropic_api_key` | `null` | Falls back to `ANTHROPIC_API_KEY` env var |

### Reranking

| Field | Default | Description |
|---|---|---|
| `rerank_enabled` | `true` | Enable composite score reranking |
| `rerank_alpha` | `0.7` | Similarity exponent (0-1). Lower = more metadata influence |
| `rerank_section_weights` | `null` | Override default section weights |
| `rerank_journal_weights` | `null` | Override default journal quartile weights |
| `oversample_multiplier` | `3` | Oversample factor before reranking |
| `oversample_topic_factor` | `5` | Additional factor for `search_topic` |

### OCR

| Field | Default | Description |
|---|---|---|
| `ocr_language` | `"eng"` | Tesseract language code for scanned pages (`"fra"`, `"deu"`, etc.). Changing requires `--force` re-index |

### OpenAlex

| Field | Default | Description |
|---|---|---|
| `openalex_email` | `null` | Email for OpenAlex polite pool (10 req/s vs 1 req/s). Falls back to `OPENALEX_EMAIL` env var |

---

## MCP tools

### Semantic search

**`search_papers`** — Passage-level semantic search. Returns matching text with surrounding context, reranked by composite score (similarity × section weight × journal weight). Supports `required_terms` for combining semantic search with exact word matching — each term must appear as a whole word in the passage.

Parameters: `query`, `top_k` (1-50), `context_chunks` (0-3), `year_min`, `year_max`, `author`, `tag`, `collection`, `chunk_types` (text/figure/table), `section_weights`, `journal_weights`, `required_terms` (list of words that must appear in passage).

**`search_topic`** — Paper-level topic search, deduplicated by document. Groups chunks by paper, scores by average and best composite relevance.

Parameters: `query`, `num_papers` (1-50), `year_min`, `year_max`, `author`, `tag`, `collection`, `chunk_types`, `section_weights`, `journal_weights`.

**`search_tables`** — Semantic search over table content (headers, cells, captions). Returns tables as markdown.

Parameters: `query`, `top_k` (1-30), `year_min`, `year_max`, `author`, `tag`, `collection`, `journal_weights`.

**`search_figures`** — Semantic search over figure captions. Returns figure metadata and paths to extracted PNGs.

Parameters: `query`, `top_k` (1-30), `year_min`, `year_max`, `author`, `tag`, `collection`.

### Boolean search

**`search_boolean`** — Exact word matching via Zotero's native full-text index. Returns papers (not passages) matching AND/OR word queries. No phrase search, no stemming.

Parameters: `query` (space-separated terms), `operator` (AND/OR), `year_min`, `year_max`.

### Context expansion

**`get_passage_context`** — Expand context around a passage from `search_papers`. For table results, pass `table_page` and `table_index` to find body text citing the table.

Parameters: `doc_id`, `chunk_index`, `window` (1-5), `table_page`, `table_index`.

### Citation graph (OpenAlex)

Requires the document to have a DOI in Zotero.

**`find_citing_papers`** — Papers that cite a given document. Parameters: `doc_id`, `limit` (1-100).

**`find_references`** — Papers a document cites. Parameters: `doc_id`, `limit` (1-100).

**`get_citation_count`** — Citation and reference counts. Parameters: `doc_id`.

### Index management

**`index_library`** — Trigger indexing from the MCP client. Parameters: `force_reindex`, `limit`, `item_key`, `title_pattern`, `no_vision`.

**`get_index_stats`** — Document/chunk/table/figure counts, section coverage, journal coverage.

**`get_reranking_config`** — Current reranking weights and valid override values.

**`get_vision_costs`** — Vision API batch usage and cost summary. Parameters: `last_n` (recent entries to show).

---

## Reranking

Search results are scored:

```
composite_score = similarity^alpha * section_weight * journal_weight
```

Default section weights:

| Section | Weight |
|---------|--------|
| results | 1.0 |
| conclusion | 1.0 |
| table | 0.9 |
| methods | 0.85 |
| abstract | 0.75 |
| background | 0.7 |
| unknown | 0.7 |
| discussion | 0.65 |
| introduction | 0.5 |
| preamble | 0.3 |
| appendix | 0.3 |
| references | 0.1 |

Default journal weights: Q1=1.0, Q2=0.85, Q3=0.65, Q4=0.45.

Override per-call via `section_weights` and `journal_weights` parameters. Set a section to 0 to exclude it. Disable reranking entirely with `"rerank_enabled": false`.

---

## Shared filter parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `author` | string | Case-insensitive substring match against author names |
| `tag` | string | Case-insensitive substring match against Zotero tags |
| `collection` | string | Case-insensitive substring match against collection names |
| `year_min` / `year_max` | int | Publication year range |
| `section_weights` | dict | Override section weights for this call |
| `journal_weights` | dict | Override journal quartile weights |
| `required_terms` | list | Exact whole-word matches required in passage (`search_papers` only) |

---

## Research agent skill

`examples/zotero-research/SKILL.md` is a ready-made Claude Code skill that wraps these
tools into a spawnable research agent — it takes a high-level research question, runs
the appropriate searches, and returns consolidated findings with citation keys. Copy it
into `.claude/skills/` (or your global skills directory) to use it.

---

## Development

### Debug viewer

`tools/debug_viewer.py` is a PyQt6 browser for inspecting the ChromaDB index — view papers, tables (rendered markdown vs PDF), figures, and individual chunks.

```bash
.venv/Scripts/python.exe tools/debug_viewer.py
```

### Tests

```bash
.venv/Scripts/python.exe -m pytest
```

Tests that make real Anthropic API calls are marked `vision_api` and excluded by
default; run them with `-m vision_api`.

`tests/stress_test_real_library.py` is the end-to-end quality gate: it pulls 10 papers
from the live Zotero library, runs the full extraction → index → search pipeline into a
temp ChromaDB, and asserts on extraction and retrieval quality. It writes
`STRESS_TEST_REPORT.md` and `_stress_test_debug.db`.

```bash
.venv/Scripts/python.exe tests/stress_test_real_library.py
```

`--vision-only` re-runs just the vision extraction against an existing
`_stress_test_debug.db`, optionally narrowed to one paper with `--paper KEY`.
