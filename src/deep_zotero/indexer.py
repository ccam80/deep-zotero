"""Indexing pipeline orchestration."""
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from .config import Config
from .zotero_client import ZoteroClient
from .pdf_processor import extract_document
from .chunker import Chunker
from .embedder import create_embedder
from .vector_store import VectorStore
from .journal_ranker import JournalRanker
from .models import ZoteroItem

logger = logging.getLogger(__name__)


def _config_hash(config: Config) -> str:
    """Hash of config values that affect indexed content.

    Changes to these values require re-indexing.
    """
    data = (
        f"{config.chunk_size}:"
        f"{config.chunk_overlap}:"
        f"{config.embedding_provider}:"
        f"{config.embedding_dimensions}:"
        f"{config.embedding_model}:"
        f"{config.ocr_language}"
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class IndexResult:
    """Outcome of indexing a single document."""
    item_key: str
    title: str
    status: str          # "indexed", "failed", "empty", "skipped"
    reason: str = ""
    n_chunks: int = 0
    n_tables: int = 0
    quality_grade: str = ""  # A/B/C/D/F quality grade per document


class Indexer:
    """
    Orchestrates the full indexing pipeline.

    Pipeline: Zotero -> PDF -> Chunks -> Embeddings -> VectorStore
    """

    def __init__(self, config: Config):
        self.config = config
        self.zotero = ZoteroClient(config.zotero_data_dir)

        self.chunker = Chunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        # Use factory to create appropriate embedder based on config
        self.embedder = create_embedder(config)
        self.store = VectorStore(config.chroma_db_path, self.embedder)
        self.journal_ranker = JournalRanker()
        self._empty_docs_path = config.chroma_db_path / "empty_docs.json"
        self._config_hash_path = config.chroma_db_path / "config_hash.txt"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            from .feature_extraction.vision_api import VisionAPI
            cost_log_path = config.chroma_db_path.parent / "vision_costs.json"
            self._vision_api = VisionAPI(
                api_key=api_key,
                cost_log_path=cost_log_path,
            )
        else:
            self._vision_api = None

    # ------------------------------------------------------------------
    # Empty-doc tracking (keyed by item_key -> pdf file hash)
    # ------------------------------------------------------------------

    def _load_empty_docs(self) -> dict[str, str]:
        """Load {item_key: pdf_hash} for docs that yielded no chunks."""
        if self._empty_docs_path.exists():
            return json.loads(self._empty_docs_path.read_text())
        return {}

    def _save_empty_docs(self, mapping: dict[str, str]) -> None:
        self._empty_docs_path.write_text(json.dumps(mapping, indent=2))

    @staticmethod
    def _pdf_hash(path: Path) -> str:
        """Fast hash of first 64 KiB of a PDF (enough to detect replacement)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(65536))
        return h.hexdigest()

    def _needs_reindex(self, item: ZoteroItem) -> tuple[bool, str]:
        """Check if a document needs (re)indexing based on PDF hash.

        Returns:
            (needs_reindex, reason) where reason is:
            - "new": Document not in index
            - "changed": PDF hash differs from stored hash
            - "no_hash": Document indexed without hash, needs reindex
            - "current": Document is up-to-date, no reindex needed
        """
        existing_meta = self.store.get_document_meta(item.item_key)
        if not existing_meta:
            return True, "new"

        stored_hash = existing_meta.get("pdf_hash")
        if not stored_hash:
            return True, "no_hash"

        current_hash = self._pdf_hash(item.pdf_path)
        if stored_hash != current_hash:
            return True, "changed"

        return False, "current"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def index_all(
        self,
        force_reindex: bool = False,
        limit: int | None = None,
        item_key: str | None = None,
        title_pattern: str | None = None,
    ) -> dict:
        """
        Index all PDFs in Zotero library.

        Args:
            force_reindex: Delete and re-index matching items
            limit: Maximum number of items to index
            item_key: If provided, only index this specific Zotero item key
            title_pattern: If provided, only index items matching this regex pattern

        Returns:
            Dict with 'results' (list[IndexResult]) and summary counts.
        """
        items = self.zotero.get_all_items_with_pdfs()
        items = [i for i in items if i.pdf_path and i.pdf_path.exists()]

        # Apply filters
        if item_key:
            items = [i for i in items if i.item_key == item_key]
            if not items:
                logger.error(f"No item found with key: {item_key}")
                return {"results": [], "indexed": 0, "failed": 0, "empty": 0, "skipped": 0, "already_indexed": 0}

        if title_pattern:
            import re
            pattern = re.compile(title_pattern, re.IGNORECASE)
            items = [i for i in items if pattern.search(i.title)]
            logger.info(f"Filtered to {len(items)} items matching pattern '{title_pattern}'")

        if limit:
            items = items[:limit]

        if force_reindex:
            existing = self.store.get_indexed_doc_ids()
            item_keys = {i.item_key for i in items}
            for doc_id in existing & item_keys:
                self.store.delete_document(doc_id)
            indexed_ids = set()
            empty_docs: dict[str, str] = {}
        else:
            indexed_ids = self.store.get_indexed_doc_ids()
            empty_docs = self._load_empty_docs()

        # Check for config mismatch
        current_hash = _config_hash(self.config)
        stored_hash = None
        if self._config_hash_path.exists():
            stored_hash = self._config_hash_path.read_text().strip()

        if stored_hash and stored_hash != current_hash and not force_reindex:
            logger.warning(
                "Config has changed since last index (chunk_size, overlap, embedding, or section settings). "
                "Run with --force to re-index, otherwise results may be inconsistent."
            )

        logger.info(f"Found {len(items)} items with PDFs")

        results: list[IndexResult] = []
        to_index: list[ZoteroItem] = []
        reindex_reasons: dict[str, str] = {}

        for item in items:
            if item.item_key in indexed_ids:
                needs_reindex, reason = self._needs_reindex(item)
                if needs_reindex:
                    self.store.delete_document(item.item_key)
                    indexed_ids.discard(item.item_key)
                    reindex_reasons[item.item_key] = reason
                    logger.info(f"Reindexing {item.item_key}: {reason}")
                else:
                    continue

            if item.item_key in empty_docs:
                current_hash = self._pdf_hash(item.pdf_path)
                if current_hash == empty_docs[item.item_key]:
                    results.append(IndexResult(
                        item.item_key, item.title, "skipped",
                        reason="no extractable text (unchanged PDF)"))
                    continue
                else:
                    del empty_docs[item.item_key]
                    reindex_reasons[item.item_key] = "changed"

            to_index.append(item)

        reindex_count = len(reindex_reasons)
        logger.info(
            f"Already indexed: {len(indexed_ids)}, "
            f"to reindex (PDF changed): {reindex_count}, "
            f"skipped (empty/unchanged): "
            f"{sum(1 for r in results if r.status == 'skipped')}, "
            f"to index: {len(to_index)}"
        )

        quality_distribution: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        aggregated_extraction_stats = {
            "total_pages": 0,
            "text_pages": 0,
            "ocr_pages": 0,
            "empty_pages": 0,
        }

        # ---- Phase 1: Extract all documents (vision specs collected but deferred) ----
        figures_dir = self.config.chroma_db_path.parent / "figures"
        doc_extractions: dict[str, tuple[ZoteroItem, object]] = {}  # item_key -> (item, extraction)

        for item in tqdm(to_index, desc="Extracting"):
            try:
                logger.debug(
                    f"Starting extraction {item.item_key}: "
                    f"title={item.title!r}, pdf={item.pdf_path}"
                )
                extraction = extract_document(
                    item.pdf_path,
                    write_images=True,
                    images_dir=figures_dir,
                    ocr_language=self.config.ocr_language,
                    vision_api=self._vision_api,
                )
                doc_extractions[item.item_key] = (item, extraction)
            except Exception as e:
                logger.error(f"Failed to extract {item.item_key}: {type(e).__name__}: {e}")
                results.append(IndexResult(
                    item.item_key, item.title, "failed",
                    reason=f"{type(e).__name__}: {e}"))

        # ---- Phase 2: Resolve vision batch (one API call for all papers) ----
        if self._vision_api and doc_extractions:
            from .pdf_processor import resolve_pending_vision
            resolve_pending_vision(
                {k: v[1] for k, v in doc_extractions.items()},
                self._vision_api,
            )

        # ---- Phase 3: Index each document (chunk, store, etc.) ----
        for item_key, (item, extraction) in doc_extractions.items():
            try:
                n_chunks, n_tables, reason, extraction_stats, quality_grade = self._index_extraction(item, extraction)

                # Aggregate extraction stats
                for key in ["total_pages", "text_pages", "ocr_pages", "empty_pages"]:
                    aggregated_extraction_stats[key] += extraction_stats.get(key, 0)

                # Track quality distribution
                if quality_grade in quality_distribution:
                    quality_distribution[quality_grade] += 1

                if n_chunks > 0:
                    results.append(IndexResult(
                        item.item_key, item.title, "indexed",
                        n_chunks=n_chunks, n_tables=n_tables,
                        quality_grade=quality_grade))
                else:
                    empty_docs[item.item_key] = self._pdf_hash(item.pdf_path)
                    results.append(IndexResult(
                        item.item_key, item.title, "empty", reason=reason,
                        quality_grade=quality_grade))
                logger.debug(f"Completed {item.item_key}: {n_chunks} chunks, {n_tables} tables, quality {quality_grade}")
            except Exception as e:
                logger.error(f"Failed to index {item.item_key}: {type(e).__name__}: {e}")
                results.append(IndexResult(
                    item.item_key, item.title, "failed",
                    reason=f"{type(e).__name__}: {e}"))

        self._save_empty_docs(empty_docs)

        counts = {
            "indexed": sum(1 for r in results if r.status == "indexed"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "empty": sum(1 for r in results if r.status == "empty"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "already_indexed": len(indexed_ids),
            "quality_distribution": quality_distribution,
            "extraction_stats": aggregated_extraction_stats,
        }

        # Save config hash after successful indexing
        if counts["indexed"] > 0 or counts["already_indexed"] > 0:
            self._config_hash_path.write_text(current_hash)

        return {"results": results, **counts}

    def _index_document_detailed(self, item: ZoteroItem) -> tuple[int, int, str, dict, str]:
        """
        Extract and index a single document (includes vision resolution).

        For batch indexing use index_all() which batches vision across all docs.
        """
        if item.pdf_path is None or not item.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for {item.item_key}")

        figures_dir = self.config.chroma_db_path.parent / "figures"
        extraction = extract_document(
            item.pdf_path,
            write_images=True,
            images_dir=figures_dir,
            ocr_language=self.config.ocr_language,
            vision_api=self._vision_api,
        )

        # Resolve vision for this single document
        if extraction.pending_vision is not None and self._vision_api:
            from .pdf_processor import resolve_pending_vision
            resolve_pending_vision({item.item_key: extraction}, self._vision_api)

        return self._index_extraction(item, extraction)

    def _index_extraction(self, item: ZoteroItem, extraction) -> tuple[int, int, str, dict, str]:
        """
        Index a pre-extracted document (vision already resolved).

        Returns:
            (n_chunks, n_tables, reason, extraction_stats, quality_grade)
        """
        if not extraction.pages:
            return 0, 0, "PDF has 0 pages (corrupt or unreadable)", extraction.stats, "F"

        total_chars = sum(len(p.markdown) for p in extraction.pages)
        quality_grade = extraction.quality_grade

        logger.debug(
            f"  Extracted {len(extraction.pages)} pages, {total_chars} chars "
            f"(text: {extraction.stats['text_pages']}, "
            f"ocr: {extraction.stats['ocr_pages']}, "
            f"empty: {extraction.stats['empty_pages']}, "
            f"quality: {quality_grade})"
        )

        if total_chars == 0:
            return 0, 0, f"{len(extraction.pages)} pages but no text", extraction.stats, quality_grade

        # Chunk using the new interface
        chunks = self.chunker.chunk(
            extraction.full_markdown,
            extraction.pages,
            extraction.sections,
        )
        if not chunks:
            return 0, 0, f"{len(extraction.pages)} pages, {total_chars} chars but no chunks created", extraction.stats, quality_grade
        logger.debug(f"  Created {len(chunks)} chunks")

        # Look up journal quartile
        journal_quartile = self.journal_ranker.lookup(item.publication)

        # Store text chunks
        doc_meta = {
            "title": item.title,
            "authors": item.authors,
            "year": item.year,
            "citation_key": item.citation_key,
            "publication": item.publication,
            "journal_quartile": journal_quartile or "",
            "doi": item.doi,
            "tags": item.tags,
            "collections": item.collections,
            "pdf_hash": self._pdf_hash(item.pdf_path),
            "quality_grade": quality_grade,
        }
        self.store.add_chunks(item.item_key, doc_meta, chunks)

        # Build reference map for table/figure placement
        from ._reference_matcher import match_references
        ref_map = match_references(extraction.full_markdown, chunks, extraction.tables, extraction.figures)

        # Enrich tables/figures with reference context.
        # Only for real captions (Table N / Figure N), not synthetic ones.
        import re
        from .pdf_processor import SYNTHETIC_CAPTION_PREFIX
        from ._reference_matcher import get_reference_context
        _TAB_NUM_RE = re.compile(r"(?:Table|Tab\.?)\s+(\d+)", re.IGNORECASE)
        _FIG_NUM_RE = re.compile(r"(?:Figure|Fig\.?)\s+(\d+)", re.IGNORECASE)
        for table in extraction.tables:
            if table.artifact_type:
                continue  # skip layout artifacts
            if table.caption and not table.caption.startswith(SYNTHETIC_CAPTION_PREFIX):
                m = _TAB_NUM_RE.search(table.caption)
                if m:
                    ctx = get_reference_context(extraction.full_markdown, chunks, ref_map, "table", int(m.group(1)))
                    table.reference_context = ctx
        for fig in extraction.figures:
            if fig.caption and not fig.caption.startswith(SYNTHETIC_CAPTION_PREFIX):
                m = _FIG_NUM_RE.search(fig.caption)
                if m:
                    ctx = get_reference_context(extraction.full_markdown, chunks, ref_map, "figure", int(m.group(1)))
                    fig.reference_context = ctx

        # Store tables if enabled (skip layout artifacts)
        n_tables = 0
        real_tables = [t for t in extraction.tables if not t.artifact_type]
        n_artifacts = len(extraction.tables) - len(real_tables)
        if real_tables:
            self.store.add_tables(item.item_key, doc_meta, real_tables, ref_map=ref_map)
            n_tables = len(real_tables)
        if n_artifacts:
            logger.debug(f"  Skipped {n_artifacts} artifact table(s)")
        logger.debug(f"  Extracted {n_tables} tables")

        # Store figures if enabled
        n_figures = 0
        if extraction.figures:
            try:
                self.store.add_figures(item.item_key, doc_meta, extraction.figures, ref_map=ref_map)
                n_figures = len(extraction.figures)
                logger.debug(f"  Extracted {n_figures} figures")
            except Exception as e:
                logger.warning(f"Figure storage failed for {item.item_key}: {e}")

        logger.debug(f"Indexed {item.item_key}: {len(chunks)} chunks, {n_tables} tables, {n_figures} figures, quality {quality_grade}")
        return len(chunks), n_tables, "", extraction.stats, quality_grade

    def index_document(self, item: ZoteroItem) -> int:
        """Index a single document. Returns number of chunks created."""
        n_chunks, _n_tables, _reason, _stats, _quality = self._index_document_detailed(item)
        return n_chunks

    def reindex_document(self, item_key: str) -> int:
        """Re-index a specific document."""
        self.store.delete_document(item_key)
        item = self.zotero.get_item(item_key)
        if item:
            return self.index_document(item)
        return 0

    def get_stats(self) -> dict:
        """Get index statistics."""
        doc_ids = self.store.get_indexed_doc_ids()
        total_chunks = self.store.count()
        return {
            "total_documents": len(doc_ids),
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(total_chunks / len(doc_ids), 1) if doc_ids else 0,
        }

    def get_library_diagnostics(self) -> dict:
        """Delegate to ZoteroClient for library-wide diagnostics."""
        return self.zotero.get_library_diagnostics()

