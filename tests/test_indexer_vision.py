"""Tests for VisionAPI construction in Indexer.__init__()."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_zotero.config import Config


def _make_config(tmp_path: Path) -> Config:
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    return Config(
        zotero_data_dir=tmp_path,
        chroma_db_path=chroma_dir,
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
        ocr_language="eng",
        openalex_email=None,
        vision_enabled=True,
        vision_model="claude-haiku-4-5-20251001",
        anthropic_api_key=None,
    )


class TestIndexerInit:
    def test_vision_api_created_with_key(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        # Set the API key via config (indexer reads config.anthropic_api_key)
        config = Config(**{**config.__dict__, "anthropic_api_key": "test-key-abc"})
        with patch("deep_zotero.indexer.ZoteroClient") as mock_zotero:
            with patch("deep_zotero.indexer.create_embedder") as mock_embedder:
                with patch("deep_zotero.indexer.VectorStore") as mock_store:
                    with patch("deep_zotero.indexer.JournalRanker"):
                        with patch(
                            "deep_zotero.feature_extraction.vision_api.VisionAPI.__init__",
                            return_value=None,
                        ) as mock_vision_init:
                            mock_zotero.return_value = MagicMock()
                            mock_embedder.return_value = MagicMock()
                            mock_store.return_value = MagicMock()

                            from deep_zotero.indexer import Indexer
                            indexer = Indexer(config)

        assert indexer._vision_api is not None
        from deep_zotero.feature_extraction.vision_api import VisionAPI
        assert isinstance(indexer._vision_api, VisionAPI)

    def test_vision_api_none_without_key(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        # anthropic_api_key=None in _make_config, so vision_api should be None
        with patch("deep_zotero.indexer.ZoteroClient") as mock_zotero:
            with patch("deep_zotero.indexer.create_embedder") as mock_embedder:
                with patch("deep_zotero.indexer.VectorStore") as mock_store:
                    with patch("deep_zotero.indexer.JournalRanker"):
                        mock_zotero.return_value = MagicMock()
                        mock_embedder.return_value = MagicMock()
                        mock_store.return_value = MagicMock()

                        from deep_zotero.indexer import Indexer
                        indexer = Indexer(config)

        assert indexer._vision_api is None

    def test_vision_api_none_when_disabled(self, tmp_path: Path) -> None:
        # vision_enabled=False should skip VisionAPI even with a key
        config = _make_config(tmp_path)
        config = Config(**{**config.__dict__, "vision_enabled": False, "anthropic_api_key": "test-key-abc"})
        with patch("deep_zotero.indexer.ZoteroClient") as mock_zotero:
            with patch("deep_zotero.indexer.create_embedder") as mock_embedder:
                with patch("deep_zotero.indexer.VectorStore") as mock_store:
                    with patch("deep_zotero.indexer.JournalRanker"):
                        mock_zotero.return_value = MagicMock()
                        mock_embedder.return_value = MagicMock()
                        mock_store.return_value = MagicMock()

                        from deep_zotero.indexer import Indexer
                        indexer = Indexer(config)

        assert indexer._vision_api is None
