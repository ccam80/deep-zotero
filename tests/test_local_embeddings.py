"""Tests for local embedding support (Feature 10).

Tests verify:
1. LocalEmbedder works without API key
2. create_embedder factory respects config
3. Dimension mismatch detection
4. Config validation for embedding providers
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from deep_zotero.config import Config
from deep_zotero.embedder import LocalEmbedder, Embedder, create_embedder
from deep_zotero.vector_store import VectorStore, EmbeddingDimensionMismatchError


def _make_config(tmp_path, **overrides):
    """Build a Config with sensible defaults, accepting overrides."""
    defaults = dict(
        zotero_data_dir=tmp_path,
        chroma_db_path=tmp_path / "chroma",
        embedding_model="gemini-embedding-001",
        embedding_dimensions=768,
        chunk_size=400,
        chunk_overlap=100,
        gemini_api_key=None,
        embedding_provider="local",
        embedding_timeout=120.0,
        embedding_max_retries=3,
        embedding_rate_limit_backoff=30.0,
        rerank_alpha=0.7,
        rerank_section_weights=None,
        rerank_journal_weights=None,
        rerank_enabled=True,
        oversample_multiplier=3,
        oversample_topic_factor=5,
        stats_sample_limit=10000,
        ocr_language="eng",
        openalex_email=None,
        vision_enabled=False,
        vision_model="claude-haiku-4-5-20251001",
        anthropic_api_key=None,
    )
    defaults.update(overrides)
    return Config(**defaults)


class TestLocalEmbedder:
    """Tests for LocalEmbedder class."""

    def test_embed_single_text(self):
        """LocalEmbedder can embed a single text."""
        embedder = LocalEmbedder()
        result = embedder.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 384  # all-MiniLM-L6-v2 dimensions

    def test_embed_multiple_texts(self):
        """LocalEmbedder can embed multiple texts."""
        embedder = LocalEmbedder()
        texts = ["First document", "Second document", "Third document"]
        result = embedder.embed(texts)

        assert len(result) == 3
        for vec in result:
            assert len(vec) == 384

    def test_embed_empty_list(self):
        """LocalEmbedder returns empty list for empty input."""
        embedder = LocalEmbedder()
        result = embedder.embed([])

        assert result == []

    def test_embed_query(self):
        """LocalEmbedder.embed_query returns single vector."""
        embedder = LocalEmbedder()
        result = embedder.embed_query("search query")

        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_documents(self):
        """LocalEmbedder.embed_documents uses same method as embed."""
        embedder = LocalEmbedder()
        texts = ["doc 1", "doc 2"]
        result = embedder.embed_documents(texts)

        assert len(result) == 2
        assert len(result[0]) == 384

    def test_dimensions_attribute(self):
        """LocalEmbedder has correct dimensions attribute."""
        embedder = LocalEmbedder()
        assert embedder.dimensions == 384

    def test_task_type_ignored(self):
        """LocalEmbedder ignores task_type (symmetric model)."""
        embedder = LocalEmbedder()
        text = ["test document"]

        # Both should produce same results (symmetric model)
        result_doc = embedder.embed(text, task_type="RETRIEVAL_DOCUMENT")
        result_query = embedder.embed(text, task_type="RETRIEVAL_QUERY")

        assert result_doc == result_query


class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_creates_local_embedder(self, tmp_path):
        """Factory creates LocalEmbedder for 'local' provider."""
        config = _make_config(tmp_path, embedding_provider="local")

        embedder = create_embedder(config)
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.dimensions == 384

    def test_creates_gemini_embedder_when_key_present(self, tmp_path, monkeypatch):
        """Factory creates Embedder for 'gemini' provider when API key is set."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

        config = _make_config(
            tmp_path,
            gemini_api_key="fake-key-for-testing",
            embedding_provider="gemini",
        )

        embedder = create_embedder(config)
        assert isinstance(embedder, Embedder)
        assert embedder.dimensions == 768

    def test_invalid_provider_raises(self, tmp_path):
        """Factory raises ValueError for invalid provider."""
        config = _make_config(tmp_path, embedding_provider="invalid_provider")

        with pytest.raises(ValueError, match="Invalid embedding_provider"):
            create_embedder(config)


class TestConfigValidation:
    """Tests for Config validation with embedding providers."""

    def test_local_provider_no_api_key_required(self, tmp_path):
        """Local provider doesn't require GEMINI_API_KEY."""
        zotero_dir = tmp_path / "zotero"
        zotero_dir.mkdir()
        (zotero_dir / "zotero.sqlite").touch()

        config = _make_config(
            tmp_path,
            zotero_data_dir=zotero_dir,
            embedding_provider="local",
            gemini_api_key=None,
        )

        errors = config.validate()
        assert not any("GEMINI_API_KEY" in e for e in errors)

    def test_gemini_provider_requires_api_key(self, tmp_path):
        """Gemini provider requires GEMINI_API_KEY."""
        zotero_dir = tmp_path / "zotero"
        zotero_dir.mkdir()
        (zotero_dir / "zotero.sqlite").touch()

        config = _make_config(
            tmp_path,
            zotero_data_dir=zotero_dir,
            embedding_provider="gemini",
            gemini_api_key=None,
        )

        errors = config.validate()
        assert any("GEMINI_API_KEY" in e for e in errors)

    def test_invalid_provider_in_validation(self, tmp_path):
        """Invalid provider caught by validation."""
        zotero_dir = tmp_path / "zotero"
        zotero_dir.mkdir()
        (zotero_dir / "zotero.sqlite").touch()

        config = _make_config(
            tmp_path,
            zotero_data_dir=zotero_dir,
            embedding_provider="openai",
        )

        errors = config.validate()
        assert any("Invalid embedding_provider" in e for e in errors)


class TestDimensionMismatch:
    """Tests for dimension mismatch detection in VectorStore."""

    def test_dimension_mismatch_raises_error(self, tmp_path):
        """VectorStore raises error when dimensions mismatch."""
        db_path = tmp_path / "chroma"

        # Create store with local embedder (384 dims)
        local_embedder = LocalEmbedder()
        store1 = VectorStore(db_path, local_embedder)

        # Add some data
        from deep_zotero.models import Chunk
        chunks = [Chunk(text="test content", page_num=1, chunk_index=0,
                       char_start=0, char_end=12, section="intro", section_confidence=1.0)]
        store1.add_chunks("doc1", {"title": "Test"}, chunks)

        # Close store to release file handles
        del store1

        # Create a mock embedder with different dimensions
        class MockGeminiEmbedder:
            dimensions = 768
            def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
                return [[0.1] * 768 for _ in texts]
            def embed_query(self, query):
                return [0.1] * 768

        # Try to open with different dimensions - should raise
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            VectorStore(db_path, MockGeminiEmbedder())

        assert "dimension mismatch" in str(exc_info.value).lower()
        assert "384" in str(exc_info.value)
        assert "768" in str(exc_info.value)

    def test_same_dimensions_ok(self, tmp_path):
        """VectorStore works when dimensions match."""
        db_path = tmp_path / "chroma"

        # Create store with local embedder
        local_embedder = LocalEmbedder()
        store1 = VectorStore(db_path, local_embedder)

        # Add some data
        from deep_zotero.models import Chunk
        chunks = [Chunk(text="test content", page_num=1, chunk_index=0,
                       char_start=0, char_end=12, section="intro", section_confidence=1.0)]
        store1.add_chunks("doc1", {"title": "Test"}, chunks)

        # Close store
        del store1

        # Reopen with same embedder type - should work
        local_embedder2 = LocalEmbedder()
        store2 = VectorStore(db_path, local_embedder2)

        assert store2.count() == 1

    def test_empty_db_any_dimension_ok(self, tmp_path):
        """Empty database accepts any embedding dimension."""
        db_path = tmp_path / "chroma"

        # Should not raise with any embedder on empty db
        local_embedder = LocalEmbedder()
        store = VectorStore(db_path, local_embedder)

        assert store.count() == 0


class TestVectorStoreWithLocalEmbeddings:
    """Integration tests for VectorStore with local embeddings."""

    def test_add_and_search(self, tmp_path):
        """Can add chunks and search with local embeddings."""
        db_path = tmp_path / "chroma"
        embedder = LocalEmbedder()
        store = VectorStore(db_path, embedder)

        from deep_zotero.models import Chunk
        chunks = [
            Chunk(text="The heart rate variability analysis showed significant changes.",
                  page_num=1, chunk_index=0, char_start=0, char_end=60,
                  section="results", section_confidence=0.9),
            Chunk(text="Methods for blood pressure measurement were validated.",
                  page_num=2, chunk_index=1, char_start=61, char_end=110,
                  section="methods", section_confidence=0.8),
        ]
        store.add_chunks("doc1", {"title": "Test Paper", "year": 2023}, chunks)

        # Search for HRV-related content
        results = store.search("heart rate variability", top_k=2)

        assert len(results) > 0
        # First result should be about HRV
        assert "heart rate" in results[0].text.lower()
