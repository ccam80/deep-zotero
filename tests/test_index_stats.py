"""Index statistics: full-collection counts and the on-disk cache."""
from pathlib import Path
from unittest.mock import MagicMock

from deep_zotero import index_stats
from deep_zotero.index_stats import (
    CACHE_FILENAME,
    IndexStatsCache,
    compute_index_stats,
    get_index_stats,
    refresh_index_stats,
)
from deep_zotero.models import Chunk
from deep_zotero.vector_store import VectorStore


def _embedder(dimensions: int = 768) -> MagicMock:
    mock = MagicMock()
    mock.dimensions = dimensions
    mock.embed = MagicMock(
        side_effect=lambda texts, **kwargs: [[0.1] * dimensions for _ in texts]
    )
    return mock


def _chunk(index: int, section: str = "methods") -> Chunk:
    return Chunk(
        text=f"chunk {index}",
        page_num=1 + index // 10,
        chunk_index=index,
        char_start=index * 10,
        char_end=index * 10 + 10,
        section=section,
        section_confidence=1.0,
    )


def _populate(store: VectorStore, docs: dict[str, int], **doc_meta) -> None:
    """Add ``count`` text chunks for each doc_id in ``docs``."""
    for doc_id, count in docs.items():
        meta = {"title": f"Doc {doc_id}", "authors": "A. Author", "year": 2024}
        meta.update(doc_meta)
        store.add_chunks(doc_id, meta, [_chunk(i) for i in range(count)])


# =============================================================================
# Full-collection counting (no sampling cap)
# =============================================================================


class TestCountsAreExact:
    def test_counts_every_chunk_beyond_one_scan_batch(self, temp_db_path: Path):
        """Totals cover the whole collection, not just the first scan batch."""
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 120, "DOC_B": 80, "DOC_C": 30})

        original = store.iter_metadatas
        store.iter_metadatas = lambda batch_size=7: original(batch_size=batch_size)

        stats = compute_index_stats(store)

        assert stats["total_chunks"] == 230
        assert stats["total_documents"] == 3
        assert sum(stats["chunk_types"].values()) == 230
        assert sum(stats["section_coverage"].values()) == 230
        assert sum(stats["journal_coverage"].values()) == 3

    def test_scan_terminates_on_exact_batch_multiple(self, temp_db_path: Path):
        """A collection sized to a whole number of batches isn't double-counted."""
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 20})

        assert len(list(store.iter_metadatas(batch_size=10))) == 20
        assert len(list(store.iter_metadatas(batch_size=20))) == 20

    def test_empty_collection(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        stats = compute_index_stats(store)

        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0
        assert stats["avg_chunks_per_doc"] == 0
        assert stats["section_coverage"] == {}

    def test_avg_chunks_per_doc(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 10, "DOC_B": 5})

        assert compute_index_stats(store)["avg_chunks_per_doc"] == 7.5

    def test_journal_coverage_counts_documents_not_chunks(self, temp_db_path: Path):
        """Quartile breakdown is per document; chunk counts must not leak in."""
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 40}, journal_quartile="Q1")
        _populate(store, {"DOC_B": 5}, journal_quartile="Q1")
        _populate(store, {"DOC_C": 12})

        stats = compute_index_stats(store)
        assert stats["journal_coverage"] == {"Q1": 2, "unknown": 1}


# =============================================================================
# Cache
# =============================================================================


class TestStatsCache:
    def test_first_call_computes_and_persists(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})

        stats = get_index_stats(store)

        assert stats["cached"] is False
        assert stats["computed_at"]
        assert (temp_db_path / CACHE_FILENAME).exists()

    def test_second_call_serves_cache(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})

        first = get_index_stats(store)
        second = get_index_stats(store)

        assert second["cached"] is True
        assert {k: v for k, v in second.items() if k != "cached"} == {
            k: v for k, v in first.items() if k != "cached"
        }

    def test_cache_is_not_rescanned(self, temp_db_path: Path, monkeypatch):
        """A cache hit must not touch the collection scan at all."""
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})
        get_index_stats(store)

        def fail(*args, **kwargs):
            raise AssertionError("cache hit should not recompute")

        monkeypatch.setattr(index_stats, "compute_index_stats", fail)
        assert get_index_stats(store)["cached"] is True

    def test_refresh_forces_recompute(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})
        get_index_stats(store)

        assert get_index_stats(store, refresh=True)["cached"] is False

    def test_cache_invalidated_when_chunk_count_changes(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})
        assert get_index_stats(store)["total_chunks"] == 12

        _populate(store, {"DOC_B": 8})
        stats = get_index_stats(store)

        assert stats["cached"] is False
        assert stats["total_chunks"] == 20
        assert stats["total_documents"] == 2

    def test_cache_invalidated_on_schema_version_change(
        self, temp_db_path: Path, monkeypatch
    ):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})
        get_index_stats(store)

        monkeypatch.setattr(index_stats, "STATS_SCHEMA_VERSION", 999)
        assert get_index_stats(store)["cached"] is False

    def test_corrupt_cache_recomputes(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})
        get_index_stats(store)

        (temp_db_path / CACHE_FILENAME).write_bytes(b"not a sqlite database")

        stats = get_index_stats(store)
        assert stats["cached"] is False
        assert stats["total_chunks"] == 12

    def test_clear_forces_recompute(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})
        get_index_stats(store)

        IndexStatsCache(temp_db_path).clear()
        assert get_index_stats(store)["cached"] is False

    def test_refresh_index_stats_updates_stored_payload(self, temp_db_path: Path):
        store = VectorStore(temp_db_path, _embedder())
        _populate(store, {"DOC_A": 12})
        get_index_stats(store)

        _populate(store, {"DOC_B": 8})
        refreshed = refresh_index_stats(store)

        assert refreshed["total_chunks"] == 20
        assert get_index_stats(store)["cached"] is True
        assert get_index_stats(store)["total_chunks"] == 20

