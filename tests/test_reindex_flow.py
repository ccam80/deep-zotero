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


def _calls(indexer: Indexer) -> list[str]:
    parent = MagicMock()
    parent.attach_mock(indexer.store.delete_document, "delete")
    parent.attach_mock(indexer._index_extraction, "index")
    return parent


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

        names = [c[0] for c in parent.mock_calls]
        assert names == ["delete", "index"]
        indexer.store.delete_document.assert_called_once_with("DOC1")
        assert result["indexed"] == 1

    def test_force_reindex_defers_delete_past_extraction(self, indexer: Indexer):
        with patch("deep_zotero.indexer.extract_document", side_effect=RuntimeError("bad pdf")):
            indexer.index_all(force_reindex=True)

        indexer.store.delete_document.assert_not_called()

    def test_force_reindex_deletes_before_store(self, indexer: Indexer):
        indexer._index_extraction = MagicMock(return_value=(5, 0, "", {}, "A"))
        parent = _calls(indexer)

        with patch("deep_zotero.indexer.extract_document", return_value=MagicMock(pending_vision=None)):
            indexer.index_all(force_reindex=True)

        assert [c[0] for c in parent.mock_calls] == ["delete", "index"]

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

        assert [c[0] for c in parent.mock_calls] == ["delete", "index"]

    def test_unknown_item_returns_zero(self, indexer: Indexer):
        indexer.zotero.get_item.return_value = None

        assert indexer.reindex_document("NOPE") == 0
        indexer.store.delete_document.assert_not_called()
