"""A document being re-indexed keeps its old copy until the new extraction is stored."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_zotero.indexer import Indexer
from deep_zotero.models import ZoteroItem


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4\nnew content\n%%EOF")
    return p


@pytest.fixture
def indexer(mock_config, pdf: Path):
    """Indexer over mocks, with one indexed doc whose stored hash is stale."""
    item = ZoteroItem(item_key="DOC1", title="Doc", authors="A", year=2024, pdf_path=pdf)

    store = MagicMock()
    store.get_indexed_doc_ids.return_value = {"DOC1"}
    store.get_document_meta.return_value = {"pdf_hash": "stale"}

    zotero = MagicMock()
    zotero.get_all_items_with_pdfs.return_value = [item]
    zotero.get_item.return_value = item

    ix = Indexer.__new__(Indexer)
    ix.config = mock_config
    ix.zotero = zotero
    ix.store = store
    ix._vision_api = None
    ix._empty_docs_path = mock_config.chroma_db_path / "empty_docs.json"
    ix._config_hash_path = mock_config.chroma_db_path / "config_hash.txt"
    return ix


@pytest.fixture(autouse=True)
def _no_stats():
    with patch("deep_zotero.indexer.refresh_index_stats", return_value={"total_documents": 0, "total_chunks": 0}):
        yield


def _calls(indexer: Indexer) -> MagicMock:
    parent = MagicMock()
    parent.attach_mock(indexer.store.snapshot_document, "snapshot")
    parent.attach_mock(indexer.store.delete_document, "delete")
    parent.attach_mock(indexer.store.restore_document, "restore")
    parent.attach_mock(indexer._index_extraction, "index")
    return parent


def _names(parent: MagicMock) -> list[str]:
    return [c[0] for c in parent.mock_calls]


class TestIndexAll:
    def test_failed_extraction_keeps_old_copy(self, indexer: Indexer):
        with patch("deep_zotero.indexer.extract_document", side_effect=RuntimeError("bad pdf")):
            result = indexer.index_all()

        indexer.store.delete_document.assert_not_called()
        assert result["failed"] == 1
        assert result["results"][0].item_key == "DOC1"

    def test_successful_extraction_deletes_then_stores(self, indexer: Indexer):
        indexer._index_extraction = MagicMock(return_value=(5, 0, "", {}, "A"))
        parent = _calls(indexer)

        with patch("deep_zotero.indexer.extract_document", return_value=MagicMock(pending_vision=None)):
            result = indexer.index_all()

        assert _names(parent) == ["snapshot", "delete", "index"]
        indexer.store.delete_document.assert_called_once_with("DOC1")
        assert result["indexed"] == 1

    def test_failed_store_restores_old_copy(self, indexer: Indexer):
        indexer._index_extraction = MagicMock(side_effect=RuntimeError("embedding down"))
        indexer.store.snapshot_document.return_value = {"ids": ["DOC1_chunk_0000"]}
        parent = _calls(indexer)

        with patch("deep_zotero.indexer.extract_document", return_value=MagicMock(pending_vision=None)):
            result = indexer.index_all()

        assert _names(parent) == ["snapshot", "delete", "index", "delete", "restore"]
        indexer.store.restore_document.assert_called_once_with({"ids": ["DOC1_chunk_0000"]})
        assert result["failed"] == 1

    def test_force_reindex_defers_delete_past_extraction(self, indexer: Indexer):
        with patch("deep_zotero.indexer.extract_document", side_effect=RuntimeError("bad pdf")):
            indexer.index_all(force_reindex=True)

        indexer.store.delete_document.assert_not_called()

    def test_force_reindex_deletes_before_store(self, indexer: Indexer):
        indexer._index_extraction = MagicMock(return_value=(5, 0, "", {}, "A"))
        parent = _calls(indexer)

        with patch("deep_zotero.indexer.extract_document", return_value=MagicMock(pending_vision=None)):
            indexer.index_all(force_reindex=True)

        assert _names(parent) == ["snapshot", "delete", "index"]

    def test_unchanged_doc_is_untouched(self, indexer: Indexer, pdf: Path):
        indexer.store.get_document_meta.return_value = {"pdf_hash": Indexer._pdf_hash(pdf)}

        with patch("deep_zotero.indexer.extract_document") as extract:
            result = indexer.index_all()

        extract.assert_not_called()
        indexer.store.delete_document.assert_not_called()
        assert result["already_indexed"] == 1


class TestReindexDocument:
    def test_failed_extraction_keeps_old_copy(self, indexer: Indexer):
        with patch("deep_zotero.indexer.extract_document", side_effect=RuntimeError("bad pdf")):
            with pytest.raises(RuntimeError):
                indexer.reindex_document("DOC1")

        indexer.store.delete_document.assert_not_called()

    def test_successful_extraction_deletes_then_stores(self, indexer: Indexer):
        indexer._index_extraction = MagicMock(return_value=(7, 0, "", {}, "A"))
        parent = _calls(indexer)

        with patch("deep_zotero.indexer.extract_document", return_value=MagicMock(pending_vision=None)):
            assert indexer.reindex_document("DOC1") == 7

        assert _names(parent) == ["snapshot", "delete", "index"]

    def test_failed_store_restores_old_copy(self, indexer: Indexer):
        indexer._index_extraction = MagicMock(side_effect=RuntimeError("embedding down"))
        parent = _calls(indexer)

        with patch("deep_zotero.indexer.extract_document", return_value=MagicMock(pending_vision=None)):
            with pytest.raises(RuntimeError):
                indexer.reindex_document("DOC1")

        assert _names(parent) == ["snapshot", "delete", "index", "delete", "restore"]


class TestSnapshotRestore:
    def test_round_trip(self, temp_db_path: Path):
        from deep_zotero.models import Chunk
        from deep_zotero.vector_store import VectorStore

        embedder = MagicMock()
        embedder.embed = MagicMock(return_value=[[0.1] * 768, [0.2] * 768])
        store = VectorStore(temp_db_path, embedder)
        chunks = [
            Chunk(text=f"chunk {i}", page_num=1, chunk_index=i, char_start=i * 10,
                  char_end=(i + 1) * 10, section="methods", section_confidence=1.0)
            for i in range(2)
        ]
        store.add_chunks("DOC1", {"title": "T", "pdf_hash": "h1"}, chunks)
        embedder.embed = MagicMock(return_value=[[0.3] * 768])
        store.add_chunks("OTHER", {"title": "O", "pdf_hash": "h2"}, chunks[:1])

        snap = store.snapshot_document("DOC1")
        store.delete_document("DOC1")
        assert store.get_document_meta("DOC1") is None
        store.restore_document(snap)

        assert store.count() == 3
        restored = store.collection.get(
            where={"doc_id": {"$eq": "DOC1"}}, include=["documents", "embeddings", "metadatas"]
        )
        assert sorted(restored["ids"]) == ["DOC1_chunk_0000", "DOC1_chunk_0001"]
        assert sorted(restored["documents"]) == ["chunk 0", "chunk 1"]
        assert all(m["pdf_hash"] == "h1" and m["section"] == "methods" for m in restored["metadatas"])
        assert [round(float(e[0]), 3) for e in restored["embeddings"]] == [0.1, 0.2]

    def test_snapshot_of_unknown_doc_restores_nothing(self, temp_db_path: Path):
        from deep_zotero.vector_store import VectorStore

        store = VectorStore(temp_db_path, MagicMock())
        store.restore_document(store.snapshot_document("NOPE"))

        assert store.count() == 0

    def test_unknown_item_returns_zero(self, indexer: Indexer):
        indexer.zotero.get_item.return_value = None

        assert indexer.reindex_document("NOPE") == 0
        indexer.store.delete_document.assert_not_called()
