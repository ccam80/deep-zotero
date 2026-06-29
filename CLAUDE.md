# deep-zotero — Project Rules

## Hard Rules

### Test Integrity

Test conditions and assertions are NEVER modified to fit the PDF corpus, perceived
limitations of extraction methods, or to make a failing test pass. If a test fails,
the code is wrong — fix the code, not the test.

### No Hard-Coded Thresholds

Every numeric threshold in the extraction pipeline MUST be adaptive — computed
from the actual data on the current page or paper. If you find yourself writing a
literal number as a threshold, STOP. Either compute it from the data, or present it
to the user for approval with alternatives.

### Never Dismiss Problems as "Inherent to the PDF" or "PyMuPDF Limitation"

PDFs are not deficient. PyMuPDF provides low-level tools (`get_text("words")`,
`get_text("dict")`, `get_drawings()`, page geometry, font metadata) that can work
around high-level function limitations. Use them.

### Performance Testing

**The stress test is the ground truth.** Run it after any change to the extraction
pipeline:

```bash
"./.venv/Scripts/python.exe" tests/stress_test_real_library.py
```

It loads 10 papers from the user's Zotero library, extracts through the full
pipeline, indexes into a temp ChromaDB, and runs ~290 assertions.

**Severity rules:**
- **MAJOR**: Orphan tables/figures, search failures, data corruption.
- **MINOR**: Section detection misses, chunk count deviations.

## Architecture

### Overview

DeepZotero is an MCP server providing deep semantic search over Zotero PDF
libraries. The pipeline:

1. **Extract** PDFs via pymupdf4llm with pymupdf-layout (text, sections, figures)
2. **Vision table extraction**: caption-driven crop → PNG → Anthropic Haiku 4.5 →
   structured JSON (headers, rows, footnotes)
3. **Chunk** text into overlapping segments with section labels
4. **Embed** via Gemini or local embeddings
5. **Store** in ChromaDB (text chunks, table markdown, figure captions)
6. **Search** via MCP tools (semantic, boolean, tables, figures, citations)

### Key files

| File | Role |
|------|------|
| `src/deep_zotero/server.py` | MCP server: all search/index/cost tools |
| `src/deep_zotero/indexer.py` | 3-phase pipeline: extract → vision batch → index |
| `src/deep_zotero/pdf_processor.py` | PDF extraction: text, sections, figures, vision table orchestration |
| `src/deep_zotero/cli.py` | CLI entry point (`deep-zotero-index`) |
| `src/deep_zotero/config.py` | Config loading from `~/.config/deep-zotero/config.json` |
| `src/deep_zotero/models.py` | Dataclasses: ExtractedTable, ExtractedFigure, SectionSpan, Chunk, etc. |
| `src/deep_zotero/chunker.py` | Text chunking with section awareness |
| `src/deep_zotero/embedder.py` | Gemini/local embedding abstraction |
| `src/deep_zotero/vector_store.py` | ChromaDB storage (chunks, tables, figures) |
| `src/deep_zotero/retriever.py` | Search result retrieval and context expansion |
| `src/deep_zotero/reranker.py` | Section/journal-weighted reranking |
| `src/deep_zotero/zotero_client.py` | Reads Zotero SQLite for items and PDF paths |
| `src/deep_zotero/journal_ranker.py` | SCImago quartile lookup |
| `src/deep_zotero/openalex_client.py` | Citation graph via OpenAlex API |
| `src/deep_zotero/section_classifier.py` | Section heading classification |
| `src/deep_zotero/orphan_recovery.py` | Recovery pass for orphan captions |
| `src/deep_zotero/_reference_matcher.py` | Maps figures/tables to citing body-text chunks |

### Vision extraction

| File | Role |
|------|------|
| `src/deep_zotero/feature_extraction/vision_api.py` | Anthropic Batch API client, cost logging, PNG rendering |
| `src/deep_zotero/feature_extraction/vision_extract.py` | System prompt, response parsing, geometry helpers (strips, re-crop) |
| `src/deep_zotero/feature_extraction/captions.py` | Unified caption detection (table + figure) |
| `src/deep_zotero/feature_extraction/debug_db.py` | Vision debug DB (agent results, run details) |
| `src/deep_zotero/feature_extraction/local_vision_api.py` | Local vision model support (vLLM/OpenAI-compatible) |
| `src/deep_zotero/feature_extraction/methods/figure_detection.py` | Figure region detection |
| `src/deep_zotero/feature_extraction/postprocessors/cell_cleaning.py` | Post-extraction cell text cleanup |

### Vision pipeline flow

1. `find_all_captions()` detects table/figure captions on each page
2. `compute_all_crops()` computes crop bbox per caption
3. `render_table_region()` renders PNG(s) — tall tables split into overlapping strips
4. Batch API submits all tables in one call with cached system prompt
5. `parse_agent_response()` extracts headers, rows, footnotes from JSON
6. Re-crop retry (max 1) if model requests tighter bounds
7. Results become `ExtractedTable` objects in the document extraction

### MCP tools

| Tool | Purpose |
|------|---------|
| `search_papers` | Passage-level semantic search with context expansion |
| `search_topic` | Find N most relevant papers, deduplicated by document |
| `search_tables` | Search table content (headers, cells, captions) |
| `search_figures` | Search figures by caption |
| `search_boolean` | Exact word matching via Zotero's full-text index |
| `get_passage_context` | Expand context around a passage or table reference |
| `find_citing_papers` | Forward citations via OpenAlex (requires DOI) |
| `find_references` | Bibliography lookup via OpenAlex (requires DOI) |
| `get_citation_count` | Quick impact check (cited_by, reference counts) |
| `get_index_stats` | Index coverage: documents, chunks, tables, figures |
| `get_reranking_config` | Current reranking weights and valid overrides |
| `get_vision_costs` | Vision API batch usage and cost summary |
| `index_library` | Trigger indexing from MCP client |

### Tools

| Tool | Purpose |
|------|---------|
| `tools/debug_viewer.py` | PyQt6 ChromaDB index browser — inspect papers, tables (md vs PDF), figures, chunks |

### Stress test

`tests/stress_test_real_library.py` — 10-paper corpus exercising extraction quality,
search accuracy, table/figure search, metadata filtering, context expansion. Produces
`STRESS_TEST_REPORT.md` and `_stress_test_debug.db`.

### Config

Default path: `~/.config/deep-zotero/config.json`

Key settings: `zotero_data_dir`, `chroma_db_path`, `embedding_provider` (gemini/local),
`vision_enabled`, `vision_model`, `anthropic_api_key`, `gemini_api_key`.
