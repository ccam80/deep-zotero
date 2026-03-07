"""MCP server with search tools."""
import os
import sys
import time
import logging
import threading
from collections import defaultdict
from dataclasses import replace
from fastmcp import FastMCP
from .config import Config
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .reranker import (
    Reranker,
    validate_section_weights,
    validate_journal_weights,
    VALID_SECTIONS,
    VALID_QUARTILES,
)
from .models import RetrievalResult

logger = logging.getLogger(__name__)

# Try to import FastMCP's error type; define fallback if not available
try:
    from fastmcp.exceptions import ToolError
except ImportError:
    class ToolError(Exception):
        """Error raised by MCP tools to signal failure to client."""
        pass


def _get_ancestor_pid():
    """
    Get the PID to monitor for parent death.

    On Windows with subprocess.Popen, there may be an intermediate process
    between the actual parent (Claude Code) and this process. We need to
    find the real parent by walking up the process tree.
    """
    if sys.platform != 'win32':
        return os.getppid()

    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL('ntdll')

    class PROCESS_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('Reserved1', ctypes.c_void_p),
            ('PebBaseAddress', ctypes.c_void_p),
            ('Reserved2', ctypes.c_void_p * 2),
            ('UniqueProcessId', wintypes.HANDLE),
            ('InheritedFromUniqueProcessId', wintypes.HANDLE),
        ]

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def get_parent_pid(pid):
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        pbi = PROCESS_BASIC_INFORMATION()
        ret_len = ctypes.c_ulong()
        status = ntdll.NtQueryInformationProcess(
            handle, 0, ctypes.byref(pbi), ctypes.sizeof(pbi), ctypes.byref(ret_len)
        )
        kernel32.CloseHandle(handle)
        if status == 0:
            return int(pbi.InheritedFromUniqueProcessId)
        return None

    # Get parent and grandparent
    parent_pid = os.getppid()
    grandparent_pid = get_parent_pid(parent_pid)

    # Return grandparent if available (skips intermediate process), else parent
    return grandparent_pid if grandparent_pid else parent_pid


def _start_parent_monitor():
    """
    Monitor parent process and exit when it dies.

    When the parent process (Claude Code) terminates, this process should
    also exit. Without this monitor, the asyncio event loop may hang
    indefinitely, leaving orphaned processes that consume CPU.
    """
    target_pid = _get_ancestor_pid()

    def monitor():
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32

            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, target_pid)

            if handle:
                # Wait for process to exit (blocks until process dies)
                INFINITE = 0xFFFFFFFF
                kernel32.WaitForSingleObject(handle, INFINITE)
                kernel32.CloseHandle(handle)
        else:
            # Unix: poll parent PID
            while True:
                time.sleep(1.0)
                try:
                    os.kill(target_pid, 0)
                except (OSError, PermissionError):
                    break

        os._exit(0)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()


# Start parent monitor before anything else
_start_parent_monitor()

mcp = FastMCP("zotero-chunk-rag")

# Lazy initialization
_retriever = None
_store = None
_reranker = None
_config = None


def _get_retriever() -> Retriever:
    global _retriever, _store, _reranker, _config
    if _retriever is None:
        _config = Config.load()
        embedder = Embedder(
            model=_config.embedding_model,
            dimensions=_config.embedding_dimensions,
            api_key=_config.gemini_api_key,
            timeout=_config.embedding_timeout,
            max_retries=_config.embedding_max_retries,
        )
        _store = VectorStore(_config.chroma_db_path, embedder)
        _retriever = Retriever(_store)
        _reranker = Reranker(alpha=_config.rerank_alpha)
    return _retriever


def _get_store() -> VectorStore:
    _get_retriever()  # Ensure initialized
    return _store


def _get_reranker() -> Reranker:
    _get_retriever()  # Ensure initialized
    return _reranker


def _stored_chunk_to_retrieval_result(chunk) -> RetrievalResult:
    """Convert a StoredChunk to RetrievalResult for reranking."""
    meta = chunk.metadata
    return RetrievalResult(
        chunk_id=chunk.id,
        text=chunk.text,
        score=chunk.score,
        doc_id=meta.get("doc_id", ""),
        doc_title=meta.get("doc_title", ""),
        authors=meta.get("authors", ""),
        year=meta.get("year"),
        page_num=meta.get("page_num", 0),
        chunk_index=meta.get("chunk_index", 0),
        citation_key=meta.get("citation_key", ""),
        publication=meta.get("publication", ""),
        section=meta.get("section", "table"),  # Tables default to "table" section
        section_confidence=meta.get("section_confidence", 1.0),
        journal_quartile=meta.get("journal_quartile"),
    )


def _build_chromadb_filters(
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict | None:
    """Build ChromaDB where clause for year range filters.

    IMPORTANT: ChromaDB only supports: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    It does NOT support substring/contains operations on metadata.
    Text-based filters (author, tag, collection) must use _apply_text_filters().

    Args:
        year_min: Minimum publication year
        year_max: Maximum publication year

    Returns:
        ChromaDB where clause dict, or None if no filters
    """
    conditions = []
    if year_min:
        conditions.append({"year": {"$gte": year_min}})
    if year_max:
        conditions.append({"year": {"$lte": year_max}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _meta_get(r, key: str, default: str = "") -> str:
    """Get a metadata field from StoredChunk (.metadata dict) or RetrievalResult (attrs)."""
    if hasattr(r, "metadata") and isinstance(r.metadata, dict):
        return r.metadata.get(key, default)
    return getattr(r, key, default)


def _apply_text_filters(
    results: list,
    author: str | None = None,
    tag: str | None = None,
    collection: str | None = None,
) -> list:
    """Apply substring-based filters in Python (post-retrieval).

    ChromaDB doesn't support substring matching, so we filter after retrieval.
    All matches are case-insensitive substrings.

    Works with both StoredChunk (metadata dict) and RetrievalResult (dataclass attrs).

    Args:
        results: List of StoredChunk or RetrievalResult objects
        author: Author name substring (case-insensitive)
        tag: Tag substring (case-insensitive)
        collection: Collection name substring (case-insensitive)

    Returns:
        Filtered list
    """
    if not author and not tag and not collection:
        return results

    author_lower = author.lower() if author else None
    tag_lower = tag.lower() if tag else None
    collection_lower = collection.lower() if collection else None

    filtered = []
    for r in results:
        if author_lower:
            authors = _meta_get(r, "authors", "").lower()
            if author_lower not in authors:
                continue

        if tag_lower:
            tags = _meta_get(r, "tags", "").lower()
            if tag_lower not in tags:
                continue

        if collection_lower:
            colls = _meta_get(r, "collections", "").lower()
            if collection_lower not in colls:
                continue

        filtered.append(r)

    return filtered


def _has_text_filters(author: str | None, tag: str | None, collection: str | None) -> bool:
    """Check if any text-based filters are active."""
    return bool(author or tag or collection)


def _result_to_dict(r) -> dict:
    """Convert RetrievalResult to API response dict.

    Expects r.composite_score to be populated by reranker.
    """
    return {
        "doc_title": r.doc_title,
        "authors": r.authors,
        "year": r.year,
        "citation_key": r.citation_key,
        "publication": r.publication,
        "page": r.page_num,
        "relevance_score": round(r.score, 3),
        "composite_score": round(r.composite_score, 3) if r.composite_score is not None else None,
        "section": r.section,
        "section_confidence": round(r.section_confidence, 2),
        "journal_quartile": r.journal_quartile,
        "passage": r.text,
        "context_before": r.context_before,
        "context_after": r.context_after,
        "full_context": r.full_context(),
        "doc_id": r.doc_id,
        "chunk_index": r.chunk_index,
    }


@mcp.tool()
def search_papers(
    query: str,
    top_k: int = 10,
    context_chunks: int = 1,
    year_min: int | None = None,
    year_max: int | None = None,
    author: str | None = None,
    tag: str | None = None,
    collection: str | None = None,
    section_weights: dict[str, float] | None = None,
    journal_weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Semantic search over research paper chunks.

    Returns relevant passages with surrounding context.

    Results are reranked by a composite score combining semantic similarity,
    document section (e.g. Results, Methods), and journal quartile (Q1-Q4).
    Pass section_weights to override default section preferences — keys are
    section labels (abstract, introduction, background, methods, results,
    discussion, conclusion, references, appendix, preamble, table, unknown),
    values are 0.0-1.0. Set a section to 0 to exclude it entirely.

    Args:
        query: Natural language search query
        top_k: Number of results (1-50)
        context_chunks: Adjacent chunks to include (0-3)
        year_min: Minimum publication year filter
        year_max: Maximum publication year filter
        author: Filter by author name (case-insensitive substring match)
        tag: Filter by Zotero tag (case-insensitive substring match)
        collection: Filter by Zotero collection name (substring match)
        section_weights: Override section weights (optional)

    Returns:
        List of results with passage text, context, and metadata
    """
    start = time.perf_counter()

    # Validate section_weights if provided
    if section_weights is not None:
        errors = validate_section_weights(section_weights)
        if errors:
            raise ToolError(f"Invalid section_weights: {'; '.join(errors)}")

    # Validate journal_weights if provided
    if journal_weights is not None:
        errors = validate_journal_weights(journal_weights)
        if errors:
            raise ToolError(f"Invalid journal_weights: {'; '.join(errors)}")

    retriever = _get_retriever()
    reranker = _get_reranker()

    # Oversample for reranking; double if text filters will reduce results
    base_fetch = min(top_k * _config.oversample_multiplier, 150)
    fetch_k = base_fetch * 2 if _has_text_filters(author, tag, collection) else base_fetch

    results = retriever.search(
        query=query,
        top_k=fetch_k,
        context_window=min(context_chunks, 3),
        filters=_build_chromadb_filters(year_min, year_max)
    )
    results = _apply_text_filters(results, author, tag, collection)

    # Rerank (or bypass if disabled)
    if _config.rerank_enabled:
        reranked = reranker.rerank(results, section_weights, journal_weights)
        top_results = reranked[:min(top_k, 50)]
    else:
        # No reranking — set composite_score equal to relevance_score
        top_results = []
        for r in results[:min(top_k, 50)]:
            result_with_score = replace(r, composite_score=r.score)
            top_results.append(result_with_score)

    logger.debug(f"search_papers: {time.perf_counter() - start:.3f}s")
    return [_result_to_dict(r) for r in top_results]


@mcp.tool()
def search_topic(
    query: str,
    num_papers: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    author: str | None = None,
    tag: str | None = None,
    collection: str | None = None,
    section_weights: dict[str, float] | None = None,
    journal_weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Find the most relevant papers for a topic, deduplicated by document.

    Searches across all chunks, then groups by paper. Each paper is scored
    by both its average composite relevance and its best single chunk.
    Results are sorted by average composite score.

    Papers are scored using composite relevance combining similarity, section,
    and journal quality. Pass section_weights to adjust section preferences.

    Args:
        query: Natural language topic description
        num_papers: Number of distinct papers to return (1-50)
        year_min: Minimum publication year filter
        year_max: Maximum publication year filter
        author: Filter by author name (case-insensitive substring match)
        tag: Filter by Zotero tag (case-insensitive substring match)
        collection: Filter by Zotero collection name (substring match)
        section_weights: Override section weights (optional)

    Returns:
        List of per-paper results with scores and best passage
    """
    start = time.perf_counter()

    # Validate section_weights if provided
    if section_weights is not None:
        errors = validate_section_weights(section_weights)
        if errors:
            raise ToolError(f"Invalid section_weights: {'; '.join(errors)}")

    # Validate journal_weights if provided
    if journal_weights is not None:
        errors = validate_journal_weights(journal_weights)
        if errors:
            raise ToolError(f"Invalid journal_weights: {'; '.join(errors)}")

    retriever = _get_retriever()
    reranker = _get_reranker()

    # Fetch more chunks than papers requested; double if text filters active
    base_fetch = min(
        num_papers * _config.oversample_topic_factor * _config.oversample_multiplier,
        600
    )
    fetch_k = base_fetch * 2 if _has_text_filters(author, tag, collection) else base_fetch

    results = retriever.search(
        query=query,
        top_k=fetch_k,
        context_window=1,
        filters=_build_chromadb_filters(year_min, year_max)
    )
    results = _apply_text_filters(results, author, tag, collection)

    # Rerank all results first (or bypass if disabled)
    if _config.rerank_enabled:
        reranked = reranker.rerank(results, section_weights, journal_weights)
    else:
        # Set composite_score = relevance_score for all results
        reranked = [replace(r, composite_score=r.score) for r in results]

    # Group by document
    by_doc: dict[str, list] = defaultdict(list)
    for r in reranked:
        by_doc[r.doc_id].append(r)

    # Score and rank papers using pre-computed composite scores
    paper_results = []
    for doc_id, hits in by_doc.items():
        # composite_score is already populated by reranker
        composite_scores = [h.composite_score for h in hits]
        avg_composite = sum(composite_scores) / len(composite_scores)

        # Best hit by composite score
        best_idx = composite_scores.index(max(composite_scores))
        best_hit = hits[best_idx]
        best_composite = composite_scores[best_idx]

        paper_results.append({
            "doc_id": doc_id,
            "doc_title": best_hit.doc_title,
            "authors": best_hit.authors,
            "year": best_hit.year,
            "citation_key": best_hit.citation_key,
            "publication": best_hit.publication,
            "journal_quartile": best_hit.journal_quartile,
            # Raw similarity scores
            "avg_score": round(sum(h.score for h in hits) / len(hits), 3),
            "best_chunk_score": round(best_hit.score, 3),
            # Composite scores
            "avg_composite_score": round(avg_composite, 3),
            "best_composite_score": round(best_composite, 3),
            "best_passage_section": best_hit.section,
            "best_passage_section_confidence": round(best_hit.section_confidence, 2),
            "num_relevant_chunks": len(hits),
            "best_passage": best_hit.text,
            "best_passage_page": best_hit.page_num,
            "best_passage_context": best_hit.full_context(),
        })

    paper_results.sort(key=lambda p: p["avg_composite_score"], reverse=True)
    logger.debug(f"search_topic: {time.perf_counter() - start:.3f}s")
    return paper_results[:num_papers]


@mcp.tool()
def search_tables(
    query: str,
    top_k: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    author: str | None = None,
    tag: str | None = None,
    collection: str | None = None,
    journal_weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Search for tables in indexed papers.

    Searches table content (headers, cells, captions) semantically.
    Returns tables as markdown with metadata.

    Results are reranked by composite score combining semantic similarity
    and journal quartile (Q1-Q4). Tables are assigned section="table" with
    default weight 0.9.

    Args:
        query: Search query describing desired table content
        top_k: Number of tables to return (1-30)
        year_min: Minimum publication year filter
        year_max: Maximum publication year filter
        author: Filter by author name (case-insensitive substring match)
        tag: Filter by Zotero tag (case-insensitive substring match)
        collection: Filter by Zotero collection name (substring match)
        journal_weights: Override journal quartile weights (optional)

    Returns:
        List of matching tables with:
        - doc_title, authors, year, citation_key: Bibliographic info
        - page: Page number where table appears
        - table_index: Index of table on page
        - caption: Table caption if detected
        - table_markdown: Full table as markdown
        - num_rows, num_cols: Table dimensions
        - relevance_score: Semantic similarity (0-1)
        - composite_score: Reranked score (similarity × section × journal)
        - doc_id: Document ID for use with get_passage_context
    """
    start = time.perf_counter()

    # Validate journal_weights if provided
    if journal_weights is not None:
        errors = validate_journal_weights(journal_weights)
        if errors:
            raise ToolError(f"Invalid journal_weights: {'; '.join(errors)}")

    top_k = max(1, min(top_k, 30))
    store = _get_store()
    reranker = _get_reranker()

    # Build filters: chunk_type=table + year range (ChromaDB-native operators only)
    type_filter = {"chunk_type": {"$eq": "table"}}
    year_filter = _build_chromadb_filters(year_min, year_max)
    filters = {"$and": [type_filter, year_filter]} if year_filter else type_filter

    # Oversample for reranking; double if text filters active
    base_fetch = min(top_k * _config.oversample_multiplier, 90)
    fetch_k = base_fetch * 2 if _has_text_filters(author, tag, collection) else base_fetch

    results = store.search(query=query, top_k=fetch_k, filters=filters)
    results = _apply_text_filters(results, author, tag, collection)

    # Apply reranking (or bypass if disabled)
    if _config.rerank_enabled:
        # Convert StoredChunk to RetrievalResult for reranking
        retrieval_results = [_stored_chunk_to_retrieval_result(r) for r in results]
        # Note: section_weights not needed - all tables have section="table"
        reranked = reranker.rerank(retrieval_results, journal_weights=journal_weights)
        top_results = reranked[:min(top_k, 30)]
    else:
        # No reranking - set composite_score = relevance_score
        retrieval_results = [_stored_chunk_to_retrieval_result(r) for r in results]
        top_results = [replace(r, composite_score=r.score) for r in retrieval_results]
        top_results = top_results[:min(top_k, 30)]

    # Build output from reranked RetrievalResult objects
    # Need to look up original StoredChunk for table-specific metadata
    result_by_id = {r.id: r for r in results}

    output = []
    for r in top_results:
        original = result_by_id.get(r.chunk_id)
        meta = original.metadata if original else {}

        output.append({
            "doc_title": r.doc_title,
            "authors": r.authors,
            "year": r.year,
            "citation_key": r.citation_key,
            "publication": r.publication,
            "journal_quartile": r.journal_quartile,
            "page": r.page_num,
            "table_index": meta.get("table_index", 0),
            "caption": meta.get("table_caption", ""),
            "table_markdown": r.text,
            "num_rows": meta.get("table_num_rows", 0),
            "num_cols": meta.get("table_num_cols", 0),
            "relevance_score": round(r.score, 3),
            "composite_score": round(r.composite_score, 3) if r.composite_score is not None else None,
            "doc_id": r.doc_id,
        })

    logger.debug(f"search_tables: {time.perf_counter() - start:.3f}s")
    return output


@mcp.tool()
def search_figures(
    query: str,
    top_k: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    author: str | None = None,
    tag: str | None = None,
    collection: str | None = None,
) -> list[dict]:
    """
    Search for figures by caption content.

    Searches figure captions semantically. Returns figures with
    their captions, page numbers, and paths to extracted images.

    Figures without detected captions are included as "orphans"
    with a generic description like "Figure on page X".

    Args:
        query: Search query for figure captions
        top_k: Number of figures to return (1-30)
        year_min: Minimum publication year filter
        year_max: Maximum publication year filter
        author: Filter by author name (case-insensitive substring match)
        tag: Filter by Zotero tag (case-insensitive substring match)
        collection: Filter by Zotero collection name (substring match)

    Returns:
        List of matching figures with:
        - doc_title, authors, year, citation_key: Bibliographic info
        - page_num: Page number where figure appears
        - figure_index: Index of figure on page
        - caption: Figure caption (empty string for orphans)
        - image_path: Path to extracted PNG image
        - relevance_score: Semantic similarity (0-1)
        - doc_id: Document ID for use with other tools
    """
    start = time.perf_counter()
    top_k = max(1, min(top_k, 30))
    store = _get_store()

    # Build filters: chunk_type=figure + year range (ChromaDB-native operators only)
    type_filter = {"chunk_type": {"$eq": "figure"}}
    year_filter = _build_chromadb_filters(year_min, year_max)
    filters = {"$and": [type_filter, year_filter]} if year_filter else type_filter

    # Oversample if text filters active
    base_fetch = min(top_k * 3, 90)
    fetch_k = base_fetch * 2 if _has_text_filters(author, tag, collection) else base_fetch

    results = store.search(query=query, top_k=fetch_k, filters=filters)
    results = _apply_text_filters(results, author, tag, collection)

    output = []
    for r in results[:top_k]:
        meta = r.metadata
        output.append({
            "doc_id": meta.get("doc_id", ""),
            "doc_title": meta.get("doc_title", ""),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "citation_key": meta.get("citation_key", ""),
            "publication": meta.get("publication", ""),
            "page_num": meta.get("page_num", 0),
            "figure_index": meta.get("figure_index", 0),
            "caption": meta.get("caption", ""),
            "image_path": meta.get("image_path", ""),
            "relevance_score": round(r.score, 3),
        })

    logger.debug(f"search_figures: {time.perf_counter() - start:.3f}s")
    return output


@mcp.tool()
def get_passage_context(
    doc_id: str,
    chunk_index: int,
    window: int = 2,
    table_page: int | None = None,
    table_index: int | None = None,
) -> dict:
    """
    Expand context around a specific passage.

    Use after search_papers to get more context.

    For table chunks (from search_tables), pass table_page and table_index
    to find the text that references the table and return that with context.

    Args:
        doc_id: Document ID from search results
        chunk_index: Chunk index from search results
        window: Chunks before/after to include (1-5)
        table_page: Page number of table (for table context lookup)
        table_index: Index of table on page (for table context lookup)
    """
    import re

    window = max(1, min(window, 5))
    store = _get_store()

    # Handle table context lookup
    if table_page is not None and table_index is not None:
        return _get_table_reference_context(store, doc_id, table_page, table_index, window)

    # Standard text chunk context
    chunks = store.get_adjacent_chunks(doc_id, chunk_index, window=window)

    if not chunks:
        raise ToolError(f"No chunks found for doc_id={doc_id}")

    # Get section and journal_quartile from center chunk
    center_chunk = next((c for c in chunks if c.metadata["chunk_index"] == chunk_index), chunks[0])

    return {
        "doc_id": doc_id,
        "doc_title": chunks[0].metadata.get("doc_title", "Unknown"),
        "citation_key": chunks[0].metadata.get("citation_key", ""),
        "section": center_chunk.metadata.get("section", "unknown"),
        "section_confidence": center_chunk.metadata.get("section_confidence", 1.0),
        "journal_quartile": center_chunk.metadata.get("journal_quartile") or None,
        "center_chunk_index": chunk_index,
        "window": window,
        "passages": [
            {
                "chunk_index": c.metadata["chunk_index"],
                "page": c.metadata["page_num"],
                "section": c.metadata.get("section", "unknown"),
                "section_confidence": c.metadata.get("section_confidence", 1.0),
                "text": c.text,
                "is_center": c.metadata["chunk_index"] == chunk_index,
            }
            for c in chunks
        ],
        "merged_text": "\n\n".join(c.text for c in chunks),
    }


def _get_table_reference_context(
    store: VectorStore,
    doc_id: str,
    table_page: int,
    table_index: int,
    window: int,
) -> dict:
    """Find text that references a specific table and return with context."""
    import re

    # Get the specific table's metadata
    table_chunk_id = f"{doc_id}_table_{table_page:04d}_{table_index:02d}"
    table_results = store.collection.get(
        ids=[table_chunk_id],
        include=["metadatas"]
    )

    if not table_results["ids"]:
        raise ToolError(f"Table not found: page={table_page}, index={table_index}")

    table_meta = table_results["metadatas"][0]
    table_caption = table_meta.get("table_caption", "")

    # Get all text chunks for this document
    text_results = store.collection.get(
        where={
            "$and": [
                {"doc_id": {"$eq": doc_id}},
                {"chunk_type": {"$eq": "text"}},
            ]
        },
        include=["documents", "metadatas"]
    )

    if not text_results["ids"]:
        # No text chunks - return table metadata only
        return {
            "doc_id": doc_id,
            "doc_title": table_meta.get("doc_title", "Unknown"),
            "citation_key": table_meta.get("citation_key", ""),
            "note": "No text chunks found for this document",
            "table_caption": table_caption,
            "table_page": table_page,
            "table_index": table_index,
            "passages": [],
            "merged_text": "",
        }

    # Extract table number from caption (e.g., "Table 1: Results" -> "1")
    table_num_match = re.search(r"Table\s*(\d+|[IVXLCDM]+)", table_caption, re.IGNORECASE)
    if table_num_match:
        table_ref = table_num_match.group(0)  # "Table 1" or "Table I"
    else:
        # Fallback: search for any table reference near this page
        table_ref = f"Table"

    # Search text chunks for reference to this table
    ref_pattern = re.compile(re.escape(table_ref), re.IGNORECASE)
    matching_chunk_idx = None

    for chunk_id, text, meta in zip(
        text_results["ids"], text_results["documents"], text_results["metadatas"]
    ):
        if ref_pattern.search(text):
            matching_chunk_idx = meta["chunk_index"]
            break

    if matching_chunk_idx is None:
        # No reference found - return table metadata with note
        return {
            "doc_id": doc_id,
            "doc_title": table_meta.get("doc_title", "Unknown"),
            "citation_key": table_meta.get("citation_key", ""),
            "note": "No text reference to this table found",
            "table_caption": table_caption,
            "table_page": table_page,
            "table_index": table_index,
            "passages": [],
            "merged_text": "",
        }

    # Found reference - get context around it
    context_chunks = store.get_adjacent_chunks(doc_id, matching_chunk_idx, window=window)
    center_chunk = next(
        (c for c in context_chunks if c.metadata["chunk_index"] == matching_chunk_idx),
        context_chunks[0] if context_chunks else None
    )

    if not center_chunk:
        raise ToolError(f"Could not retrieve context for chunk {matching_chunk_idx}")

    return {
        "doc_id": doc_id,
        "doc_title": center_chunk.metadata.get("doc_title", "Unknown"),
        "citation_key": center_chunk.metadata.get("citation_key", ""),
        "table_caption": table_caption,
        "table_page": table_page,
        "table_index": table_index,
        "reference_found_in_chunk": matching_chunk_idx,
        "section": center_chunk.metadata.get("section", "unknown"),
        "section_confidence": center_chunk.metadata.get("section_confidence", 1.0),
        "center_chunk_index": matching_chunk_idx,
        "window": window,
        "passages": [
            {
                "chunk_index": c.metadata["chunk_index"],
                "page": c.metadata["page_num"],
                "section": c.metadata.get("section", "unknown"),
                "text": c.text,
                "is_center": c.metadata["chunk_index"] == matching_chunk_idx,
            }
            for c in context_chunks
        ],
        "merged_text": "\n\n".join(c.text for c in context_chunks),
    }


@mcp.tool()
def get_index_stats() -> dict:
    """Get statistics about the indexed collection."""
    _get_retriever()  # Ensure initialized
    store = _get_store()
    doc_ids = store.get_indexed_doc_ids()
    total_chunks = store.count()

    # Get section, journal, and chunk type coverage from a sample of chunks
    # (Getting all chunks would be expensive for large collections)
    sample = store.collection.get(limit=_config.stats_sample_limit, include=["metadatas"])

    section_counts: dict[str, int] = defaultdict(int)
    journal_doc_quartiles: dict[str, str] = {}  # doc_id -> quartile
    chunk_type_counts: dict[str, int] = defaultdict(int)

    if sample["metadatas"]:
        for meta in sample["metadatas"]:
            section = meta.get("section", "unknown")
            section_counts[section] += 1

            chunk_type = meta.get("chunk_type", "text")
            chunk_type_counts[chunk_type] += 1

            doc_id = meta.get("doc_id", "")
            quartile = meta.get("journal_quartile", "")
            if doc_id and doc_id not in journal_doc_quartiles:
                journal_doc_quartiles[doc_id] = quartile

    # Count documents per quartile
    journal_counts: dict[str, int] = defaultdict(int)
    for quartile in journal_doc_quartiles.values():
        key = quartile if quartile else "unknown"
        journal_counts[key] += 1

    return {
        "total_documents": len(doc_ids),
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": round(total_chunks / len(doc_ids), 1) if doc_ids else 0,
        "section_coverage": dict(section_counts),
        "journal_coverage": dict(journal_counts),
        "chunk_types": dict(chunk_type_counts),
    }


@mcp.tool()
def get_reranking_config() -> dict:
    """
    Get current reranking configuration.

    Returns section weights, journal quartile weights, alpha exponent,
    and valid section names for use with section_weights parameter.
    """
    _get_retriever()  # Ensure initialized
    reranker = _get_reranker()

    return {
        "enabled": _config.rerank_enabled,
        "alpha": reranker.alpha,
        "section_weights": reranker.default_section_weights,
        "journal_weights": {
            k if k is not None else "unknown": v
            for k, v in reranker.quartile_weights.items()
            if k != ""  # Skip the empty string duplicate
        },
        "valid_sections": sorted(VALID_SECTIONS),
        "valid_quartiles": sorted(VALID_QUARTILES),
        "oversample_multiplier": _config.oversample_multiplier,
    }


# =============================================================================
# Boolean Full-Text Search (Feature 3)
# =============================================================================


@mcp.tool()
def search_boolean(
    query: str,
    operator: str = "AND",
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[dict]:
    """
    Boolean full-text search using Zotero's native word index.

    Use for exact word matching with AND/OR logic. Unlike semantic search,
    this finds exact word matches only (no synonyms or similar meaning).

    This searches the full text of PDFs that Zotero has indexed. Words are
    tokenized by Zotero's indexer, so punctuation and hyphenation affect
    matching (e.g., "heart-rate" is two words: "heart" and "rate").

    Limitations:
    - No phrase search ("heart rate" searches for both words, not the phrase)
    - No stemming ("running" won't match "run")
    - Requires Zotero to have indexed the PDFs

    Args:
        query: Space-separated search terms (case-insensitive)
        operator: "AND" (all terms required) or "OR" (any term matches)
        year_min: Minimum publication year filter
        year_max: Maximum publication year filter

    Returns:
        List of matching papers with metadata (no passages - use search_papers
        for passage retrieval on specific papers)
    """
    from .zotero_client import ZoteroClient

    # Get config lazily
    global _config
    if _config is None:
        _config = Config.load()

    zotero = ZoteroClient(_config.zotero_data_dir)
    matching_keys = zotero.search_fulltext(query, operator)

    if not matching_keys:
        return []

    # Get metadata for matching items
    all_items = zotero.get_all_items_with_pdfs()
    items_by_key = {i.item_key: i for i in all_items}

    results = []
    for key in matching_keys:
        item = items_by_key.get(key)
        if not item:
            continue

        # Apply year filters
        if year_min and (item.year is None or item.year < year_min):
            continue
        if year_max and (item.year is None or item.year > year_max):
            continue

        results.append({
            "item_key": item.item_key,
            "title": item.title,
            "authors": item.authors,
            "year": item.year,
            "publication": item.publication,
            "citation_key": item.citation_key,
            "tags": item.tags,
            "collections": item.collections,
            "doi": item.doi,
        })

    # Sort by year descending
    results.sort(key=lambda x: x.get("year") or 0, reverse=True)
    return results


# =============================================================================
# Citation Graph (Feature 9 - OpenAlex)
# =============================================================================


@mcp.tool()
def find_citing_papers(doc_id: str, limit: int = 20) -> list[dict]:
    """
    Find papers that cite a given document.

    Requires the document to have a DOI. Uses OpenAlex API for citation data.
    Rate-limited to 1 request/second (or 10/second if openalex_email configured).

    Args:
        doc_id: Document ID (Zotero item key) from search results
        limit: Maximum number of citing papers to return (1-100)

    Returns:
        List of citing papers with title, authors, year, DOI, and citation count
    """
    store = _get_store()
    meta = store.get_document_meta(doc_id)
    if not meta:
        raise ToolError(f"Document not found: {doc_id}")

    doi = meta.get("doi")
    if not doi:
        raise ToolError("Document has no DOI - citation lookup unavailable")

    from .openalex_client import OpenAlexClient

    global _config
    if _config is None:
        _config = Config.load()

    client = OpenAlexClient(email=_config.openalex_email)

    work = client.get_work_by_doi(doi)
    if not work:
        raise ToolError(f"Paper not found in OpenAlex: {doi}")

    citing = client.get_citing_works(work.openalex_id, limit)

    return [client.format_work(w) for w in citing]


@mcp.tool()
def find_references(doc_id: str, limit: int = 50) -> list[dict]:
    """
    Find papers that a document references (its bibliography).

    Requires the document to have a DOI. Uses OpenAlex API.
    Rate-limited to 1 request/second (or 10/second if openalex_email configured).

    Args:
        doc_id: Document ID (Zotero item key) from search results
        limit: Maximum number of references to return (1-100)

    Returns:
        List of referenced papers with title, authors, year, DOI, and citation count
    """
    store = _get_store()
    meta = store.get_document_meta(doc_id)
    if not meta:
        raise ToolError(f"Document not found: {doc_id}")

    doi = meta.get("doi")
    if not doi:
        raise ToolError("Document has no DOI - reference lookup unavailable")

    from .openalex_client import OpenAlexClient

    global _config
    if _config is None:
        _config = Config.load()

    client = OpenAlexClient(email=_config.openalex_email)

    work = client.get_work_by_doi(doi)
    if not work:
        raise ToolError(f"Paper not found in OpenAlex: {doi}")

    references = client.get_references(work.openalex_id, limit)

    return [client.format_work(w) for w in references]


@mcp.tool()
def get_citation_count(doc_id: str) -> dict:
    """
    Get citation count and reference count for a document.

    Requires the document to have a DOI. Uses OpenAlex API.

    Args:
        doc_id: Document ID (Zotero item key) from search results

    Returns:
        Dict with cited_by_count and reference_count
    """
    store = _get_store()
    meta = store.get_document_meta(doc_id)
    if not meta:
        raise ToolError(f"Document not found: {doc_id}")

    doi = meta.get("doi")
    if not doi:
        raise ToolError("Document has no DOI - citation lookup unavailable")

    from .openalex_client import OpenAlexClient

    global _config
    if _config is None:
        _config = Config.load()

    client = OpenAlexClient(email=_config.openalex_email)

    work = client.get_work_by_doi(doi)
    if not work:
        raise ToolError(f"Paper not found in OpenAlex: {doi}")

    return {
        "doc_id": doc_id,
        "doi": doi,
        "openalex_id": work.openalex_id,
        "cited_by_count": work.cited_by_count,
        "reference_count": len(work.references),
    }


if __name__ == "__main__":
    mcp.run()
