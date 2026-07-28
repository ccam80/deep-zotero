"""Targeted tests for citation-key source merging and metadata backfill."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deep_zotero.vector_store import VectorStore
from deep_zotero.zotero_client import ZoteroClient


def _create_native_db(data_dir: Path, values: dict[str, str], field_id: int = 731) -> None:
    """Create only the Zotero EAV schema needed by citation-key loading."""
    conn = sqlite3.connect(data_dir / "zotero.sqlite")
    conn.executescript("""
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, "key" TEXT NOT NULL);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT NOT NULL);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
    """)
    conn.execute(
        "INSERT INTO fields (fieldID, fieldName) VALUES (?, 'citationKey')",
        (field_id,),
    )
    for index, (item_key, citation_key) in enumerate(values.items(), start=1):
        conn.execute("INSERT INTO items (itemID, \"key\") VALUES (?, ?)", (index, item_key))
        conn.execute(
            "INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)",
            (index, citation_key),
        )
        conn.execute(
            "INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
            (index, field_id, index),
        )
    conn.commit()
    conn.close()


def _create_legacy_db(data_dir: Path, values: dict[str, str]) -> None:
    conn = sqlite3.connect(data_dir / "better-bibtex.sqlite")
    conn.execute("CREATE TABLE citationkey (itemKey TEXT, citationKey TEXT)")
    conn.executemany(
        "INSERT INTO citationkey (itemKey, citationKey) VALUES (?, ?)",
        values.items(),
    )
    conn.commit()
    conn.close()


class TestCitationKeySources:
    def test_native_keys_load_while_database_has_an_exclusive_lock(self, tmp_path):
        """Immutable read-only access must not wait on Zotero's live DB lock."""
        _create_native_db(tmp_path, {"LOCKED01": "LockedKey"})
        lock = sqlite3.connect(tmp_path / "zotero.sqlite")
        lock.execute("BEGIN EXCLUSIVE")
        try:
            keys = ZoteroClient(tmp_path)._load_native_citation_keys()
        finally:
            lock.rollback()
            lock.close()

        assert keys == {"LOCKED01": "LockedKey"}

    def test_loads_native_key_by_field_name_not_field_id(self, tmp_path):
        _create_native_db(tmp_path, {"NATIVE01": "Smith2024"}, field_id=9876)

        keys = ZoteroClient(tmp_path)._load_citation_keys()

        assert keys == {"NATIVE01": "Smith2024"}

    def test_legacy_sidecar_is_fallback_when_native_key_missing(self, tmp_path):
        _create_native_db(tmp_path, {})
        _create_legacy_db(tmp_path, {"LEGACY01": "Jones2020"})

        keys = ZoteroClient(tmp_path)._load_citation_keys()

        assert keys == {"LEGACY01": "Jones2020"}

    def test_native_precedence_with_partial_legacy_mapping(self, tmp_path):
        _create_native_db(
            tmp_path,
            {"BOTH0001": "NativeKey", "NATIVE02": "NativeOnly"},
        )
        _create_legacy_db(
            tmp_path,
            {"BOTH0001": "OldLegacyKey", "LEGACY02": "LegacyOnly"},
        )

        keys = ZoteroClient(tmp_path)._load_citation_keys()

        assert keys == {
            "BOTH0001": "NativeKey",
            "NATIVE02": "NativeOnly",
            "LEGACY02": "LegacyOnly",
        }


@pytest.fixture
def metadata_store(tmp_path):
    embedder = Mock()
    embedder.dimensions = 3
    store = VectorStore(tmp_path / "chroma", embedder)
    ids = [
        "DOC00001_chunk_0000",
        "DOC00001_table_0001_00",
        "DOC00001_fig_001_00",
        "DOC00002_chunk_0000",
        "DOC00003_chunk_0000",
    ]
    store.collection.add(
        ids=ids,
        documents=["text", "table", "figure", "same", "missing"],
        embeddings=[[0.1, 0.2, 0.3]] * len(ids),
        metadatas=[
            {"doc_id": "DOC00001", "chunk_type": "text", "citation_key": ""},
            {"doc_id": "DOC00001", "chunk_type": "table", "citation_key": "OldKey"},
            {"doc_id": "DOC00001", "chunk_type": "figure", "citation_key": "NewKey"},
            {"doc_id": "DOC00002", "chunk_type": "text", "citation_key": "SameKey"},
            {"doc_id": "DOC00003", "chunk_type": "text", "citation_key": ""},
        ],
    )
    return store, embedder


class TestCitationKeyMetadataRefresh:
    def test_dry_run_reports_without_mutation(self, metadata_store):
        store, embedder = metadata_store

        report = store.refresh_citation_keys(
            {"DOC00001": "NewKey", "DOC00002": "SameKey"},
            dry_run=True,
        )

        assert report == {
            "dry_run": True,
            "documents_examined": 3,
            "documents_changed": 1,
            "documents_unchanged": 1,
            "documents_missing": 1,
            "documents_failed": 0,
            "records_examined": 5,
            "records_changed": 2,
            "records_unchanged": 2,
            "records_missing": 1,
            "records_failed": 0,
            "failures": [],
        }
        stored = store.collection.get(
            where={"doc_id": {"$eq": "DOC00001"}},
            include=["metadatas"],
        )
        assert [m["citation_key"] for m in stored["metadatas"]] == ["", "OldKey", "NewKey"]
        embedder.embed.assert_not_called()

    def test_mutates_all_changed_chunk_types_without_embedding(self, metadata_store):
        store, embedder = metadata_store

        report = store.refresh_citation_keys(
            {"DOC00001": "NewKey", "DOC00002": "SameKey"},
            dry_run=False,
        )

        stored = store.collection.get(
            where={"doc_id": {"$eq": "DOC00001"}},
            include=["metadatas", "embeddings"],
        )
        assert {m["chunk_type"] for m in stored["metadatas"]} == {"text", "table", "figure"}
        assert {m["citation_key"] for m in stored["metadatas"]} == {"NewKey"}
        assert stored["embeddings"] is not None
        assert report["documents_changed"] == 1
        assert report["records_changed"] == 2
        embedder.embed.assert_not_called()

    def test_update_failure_is_reported_and_does_not_claim_changes(self, metadata_store, monkeypatch):
        store, _ = metadata_store
        original_update = store.collection.update

        def fail_doc_one(*, ids, metadatas):
            if any(record_id.startswith("DOC00001_") for record_id in ids):
                raise RuntimeError("write blocked")
            return original_update(ids=ids, metadatas=metadatas)

        monkeypatch.setattr(store.collection, "update", fail_doc_one)

        report = store.refresh_citation_keys(
            {"DOC00001": "NewKey", "DOC00002": "SameKey"},
            dry_run=False,
        )

        assert report["documents_failed"] == 1
        assert report["records_failed"] == 2
        assert report["documents_changed"] == 0
        assert report["records_changed"] == 0
        assert report["failures"] == [
            {"doc_id": "DOC00001", "error": "RuntimeError: write blocked"}
        ]


def test_mcp_refresh_uses_metadata_path_without_extraction(monkeypatch, tmp_path):
    """The public MCP workflow must not enter the PDF indexing pipeline."""
    from deep_zotero import pdf_processor, server, zotero_client

    fake_store = Mock()
    fake_store.refresh_citation_keys.return_value = {"dry_run": False}
    fake_zotero = Mock()
    fake_zotero.get_citation_keys.return_value = {"DOC00001": "Smith2024"}
    extraction = Mock(side_effect=AssertionError("PDF extraction must not run"))

    monkeypatch.setattr(
        server,
        "_config",
        SimpleNamespace(zotero_data_dir=tmp_path, validate=lambda: []),
    )
    monkeypatch.setattr(server, "_get_store", lambda: fake_store)
    monkeypatch.setattr(zotero_client, "ZoteroClient", lambda _: fake_zotero)
    monkeypatch.setattr(pdf_processor, "extract_document", extraction)

    result = server.refresh_citation_keys(dry_run=False)

    assert result == {"dry_run": False}
    fake_store.refresh_citation_keys.assert_called_once_with(
        {"DOC00001": "Smith2024"},
        dry_run=False,
    )
    extraction.assert_not_called()
