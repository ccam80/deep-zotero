# DeepZotero

Semantic search over a Zotero library. PDFs are extracted (text, tables, figures), chunked, embedded, and stored in ChromaDB. An MCP server exposes the index to Claude Code (or any MCP client) as 10 tools for semantic and exact-word search over text, tables and figures, context expansion, citation graph lookup, indexing, and cost tracking.

## What it extracts

- **Text** — section-aware chunks with overlap, classified by document section (abstract, methods, results, etc.)
- **Tables** — vision-based extraction via Claude Haiku 4.5. Each table is rendered to PNG and transcribed to structured markdown (headers, rows, footnotes). Table extraction is vision-only: with vision disabled, or without an Anthropic key, no tables are extracted.
- **Figures** — detected with captions, extracted as PNGs, searchable by caption text.

## Requirements

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/app/apikey) for embeddings (unless using `embedding_provider: "local"`)
- An [Anthropic API key](https://console.anthropic.com/) for vision-based table extraction (optional but recommended)
- **Zotero 8** with PDFs in `storage/`. Citation keys are read from Zotero's native `citationKey` field, which earlier versions do not have — on Zotero 7 every citation key comes back empty.
- **Tesseract-OCR** — only needed to OCR scanned / image-only PDF pages. Install [Tesseract](https://github.com/tesseract-ocr/tesseract) with the language data you need, then set the `TESSDATA_PREFIX` environment variable to its `tessdata` directory (e.g. `C:\Program Files\Tesseract-OCR\tessdata`). PyMuPDF locates the OCR data via that variable; without it, scanned pages are skipped (`"OCR disabled because Tesseract language data not found."`). Text-based PDFs do not need Tesseract.

## Install as a Claude Code plugin

Requires [uv](https://docs.astral.sh/uv/) on `PATH`. The repo is its own marketplace:

```
/plugin marketplace add ccam80/deep-zotero
/plugin install deep-zotero
```

The server launches through `uvx`, which fetches the pinned `deep-zotero` wheel from PyPI and caches it.

In the environment Claude Code starts from, set `DEEP_ZOTERO_DATA_DIR` (the Zotero data directory holding `zotero.sqlite` and `storage/`), `DEEP_ZOTERO_CHROMA_PATH` (where the index lives), `GEMINI_API_KEY` (embeddings) and `ANTHROPIC_API_KEY` (vision table extraction during indexing).

Index the library once before searching: `deep-zotero-index -v`.

`/deep-zotero:install` walks a coding agent through the whole setup: variables, Tesseract, and the first index.

## Setup

### 1. Configuration

The four environment variables above are the whole configuration. Every other setting has a sensible default.

To override more than those four, write a JSON config and point `DEEP_ZOTERO_CONFIG` at it (or pass `--config PATH` to the CLI):

```json
{
    "zotero_data_dir": "~/Zotero",
    "chroma_db_path": "~/.local/share/deep-zotero/chroma",
    "gemini_api_key": "YOUR_GEMINI_KEY",
    "anthropic_api_key": "YOUR_ANTHROPIC_KEY"
}
```

A config file takes precedence over the environment wherever both supply a setting. See [Configuration reference](#configuration-reference) for every field.

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

### 4. Check it works

Call `get_index_stats` from Claude Code. It should report the documents and chunks just indexed.

If the tools are missing, the server did not start: confirm `uv` is on `PATH` and that the environment variables are visible to Claude Code's process.

For scanned-page OCR, `TESSDATA_PREFIX` (see [Requirements](#requirements)) must be set in that same environment.

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

Parameters: `query` (optional when `required_terms` is given), `top_k` (1-50), `context_chunks` (0-3), `year_min`, `year_max`, `author`, `tag`, `collection`, `chunk_types` (text/figure/table), `sections`, `journal_quartiles`, `section_weights`, `journal_weights`, `required_terms` (words that must appear as whole words), `terms_operator` (AND/OR).

**`search_topic`** — Paper-level topic search, deduplicated by document. Groups chunks by paper, scores by average and best composite relevance.

Parameters: `query`, `num_papers` (1-50), `year_min`, `year_max`, `author`, `tag`, `collection`, `chunk_types`, `sections`, `journal_quartiles`, `section_weights`, `journal_weights`.

### Filtering

```python
search_papers(query="baroreflex sensitivity", author="Olufsen", journal_quartiles=["Q1"])
search_papers(required_terms=["SDNN"], sections=["results"], year_min=1991, year_max=1995)
```

| Parameter | Applied | Match |
|---|---|---|
| `year_min`, `year_max`, `chunk_types`, `sections`, `journal_quartiles` | during retrieval | exact |
| `author`, `tag`, `collection` | after retrieval | case-insensitive substring |
| `section_weights`, `journal_weights` | during reranking, so only with `query` | reorders, excludes nothing |

Valid `sections`: `abstract`, `introduction`, `background`, `methods`, `results`, `discussion`, `conclusion`, `references`, `appendix`, `preamble`, `table`, `figure`, `unknown`.

Valid `journal_quartiles`: `Q1`, `Q2`, `Q3`, `Q4`, and `unknown` for journals with no quartile.

### Tables and figures

`search_papers` covers text, tables and figures. Every result names its
`chunk_type`; narrow with `chunk_types`:

```python
search_papers("impedance measurement", chunk_types=["table"])
```

Table results add `table_index`, `caption`, `num_rows` and `num_cols`, with the
table markdown in `passage`. Figure results add `figure_index`, `caption` and
`image_path` (the extracted PNG), with the caption in `passage`.

Table and figure chunks carry `section` values of `table` and `figure` rather
than the section they appeared in. Both weigh 1.0 in reranking.

### Exact word matching

`required_terms` lists words that must appear in the passage as whole words,
case-insensitively — `heart` matches `Heart` but not `hearth`. `terms_operator`
is `AND` (default) or `OR`.

Terms constrain the search itself, so a passage containing a rare acronym is
found even when semantic similarity would not rank it:

```python
search_papers("autonomic regulation", required_terms=["SDNN"])
```

Omit `query` to retrieve every matching passage in the index, unranked, with no
embedding call:

```python
search_papers(required_terms=["propranolol", "SDNN"], terms_operator="AND")
```

At least one of `query` or `required_terms` is required. `section_weights` and
`journal_weights` affect ranking only and are ignored when `query` is omitted.
No phrase search, no stemming.

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

**`get_index_stats`** — Document/chunk/table/figure counts, section coverage, journal coverage. Counts cover the entire collection. The result is cached in `index_stats.sqlite` next to the Chroma database and refreshed at the end of every indexing run; pass `refresh: true` to force a recount.

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

## Releasing

`python tools/release.py --bump patch` (or an explicit `X.Y.Z`) rewrites the version in `pyproject.toml` and `plugin/.claude-plugin/plugin.json`, including the pinned package the plugin launches, then commits and tags. It refuses a dirty tree, a non-`main` branch, and an existing tag.

Pushing the tag runs `.github/workflows/publish.yml`, which re-checks that tag, both manifests and the pin agree before publishing to PyPI.
