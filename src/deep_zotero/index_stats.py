"""Statistics over the ChromaDB index, computed from a full scan and cached."""
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_store import VectorStore

logger = logging.getLogger(__name__)

CACHE_FILENAME = "index_stats.sqlite"

# Bump when the shape of the computed payload changes.
STATS_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""


class IndexStatsCache:
    """SQLite-backed cache for a single computed statistics payload."""

    def __init__(self, db_dir: Path):
        self.db_dir = Path(db_dir)
        self.path = self.db_dir / CACHE_FILENAME

    def _connect(self) -> sqlite3.Connection:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute(_SCHEMA)
        return conn

    def read(self, expected_chunk_count: int) -> dict | None:
        """Return the cached payload, or None if it is absent or stale."""
        if not self.path.exists():
            return None
        try:
            conn = self._connect()
        except sqlite3.Error as e:
            logger.warning(f"Index stats cache unreadable ({e}); recomputing")
            return None
        try:
            row = conn.execute(
                "SELECT schema_version, chunk_count, computed_at, payload "
                "FROM index_stats WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as e:
            logger.warning(f"Index stats cache unreadable ({e}); recomputing")
            return None
        finally:
            conn.close()

        if row is None:
            return None
        if row["schema_version"] != STATS_SCHEMA_VERSION:
            return None
        if row["chunk_count"] != expected_chunk_count:
            return None
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            logger.warning("Index stats cache payload corrupt; recomputing")
            return None
        payload["computed_at"] = row["computed_at"]
        return payload

    def write(self, stats: dict, chunk_count: int) -> str:
        """Persist ``stats``. Returns the recorded timestamp."""
        computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {k: v for k, v in stats.items() if k != "computed_at"}
        try:
            conn = self._connect()
        except sqlite3.Error as e:
            logger.warning(f"Could not open index stats cache for writing: {e}")
            return computed_at
        try:
            with conn:
                conn.execute(
                    "INSERT INTO index_stats "
                    "(id, schema_version, chunk_count, computed_at, payload) "
                    "VALUES (1, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "schema_version=excluded.schema_version, "
                    "chunk_count=excluded.chunk_count, "
                    "computed_at=excluded.computed_at, "
                    "payload=excluded.payload",
                    (
                        STATS_SCHEMA_VERSION,
                        chunk_count,
                        computed_at,
                        json.dumps(payload),
                    ),
                )
        except sqlite3.Error as e:
            logger.warning(f"Could not write index stats cache: {e}")
        finally:
            conn.close()
        return computed_at

    def clear(self) -> None:
        """Drop the cached payload, forcing the next read to recompute."""
        if not self.path.exists():
            return
        try:
            conn = self._connect()
        except sqlite3.Error:
            return
        try:
            with conn:
                conn.execute("DELETE FROM index_stats")
        except sqlite3.Error as e:
            logger.warning(f"Could not clear index stats cache: {e}")
        finally:
            conn.close()


def _by_count_desc(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def compute_index_stats(store: "VectorStore") -> dict:
    """Compute statistics from a full scan of the collection."""
    section_counts: dict[str, int] = defaultdict(int)
    chunk_type_counts: dict[str, int] = defaultdict(int)
    doc_quartiles: dict[str, str] = {}
    total_chunks = 0

    for meta in store.iter_metadatas():
        total_chunks += 1
        section_counts[meta.get("section") or "unknown"] += 1
        chunk_type_counts[meta.get("chunk_type") or "text"] += 1

        doc_id = meta.get("doc_id") or ""
        if doc_id and doc_id not in doc_quartiles:
            doc_quartiles[doc_id] = meta.get("journal_quartile") or ""

    journal_counts: dict[str, int] = defaultdict(int)
    for quartile in doc_quartiles.values():
        journal_counts[quartile or "unknown"] += 1

    total_documents = len(doc_quartiles)
    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": (
            round(total_chunks / total_documents, 1) if total_documents else 0
        ),
        "section_coverage": _by_count_desc(section_counts),
        "journal_coverage": _by_count_desc(journal_counts),
        "chunk_types": _by_count_desc(chunk_type_counts),
    }


def get_index_stats(store: "VectorStore", refresh: bool = False) -> dict:
    """Return index statistics, plus ``computed_at`` and ``cached``."""
    cache = IndexStatsCache(store.db_path)
    chunk_count = store.count()

    if not refresh:
        cached = cache.read(chunk_count)
        if cached is not None:
            cached["cached"] = True
            return cached

    stats = compute_index_stats(store)
    computed_at = cache.write(stats, stats["total_chunks"])
    stats["computed_at"] = computed_at
    stats["cached"] = False
    return stats


def refresh_index_stats(store: "VectorStore") -> dict:
    """Recompute and persist statistics."""
    return get_index_stats(store, refresh=True)
