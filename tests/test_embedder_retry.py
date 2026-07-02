"""Retry/backoff behaviour of the Gemini Embedder.

Focus: rate-limit (HTTP 429 / RESOURCE_EXHAUSTED) responses must use a long,
fixed backoff that spans the per-minute quota window, while other transient
errors keep the fast exponential backoff.
"""
from unittest import mock

import pytest

from deep_zotero.embedder import Embedder, EmbeddingError


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeResponse:
    def __init__(self, n):
        self.embeddings = [_FakeEmbedding([0.0] * 768) for _ in range(n)]


class _RateLimitError(Exception):
    """Mimics google.genai ClientError for a 429 quota response."""
    code = 429

    def __str__(self):
        return (
            "429 RESOURCE_EXHAUSTED. You exceeded your current quota, "
            "please check your plan and billing details."
        )


def _bare_embedder(**overrides):
    """Build an Embedder without constructing a real genai client."""
    emb = Embedder.__new__(Embedder)
    emb.model = "gemini-embedding-001"
    emb.dimensions = 768
    emb.timeout = 120.0
    emb.max_retries = 3
    emb.rate_limit_backoff = 30.0
    emb.client = mock.MagicMock()
    for k, v in overrides.items():
        setattr(emb, k, v)
    return emb


@pytest.fixture
def captured_sleeps(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("deep_zotero.embedder.time.sleep", lambda s: sleeps.append(s))
    return sleeps


def test_rate_limit_recovers_with_30s_backoff(captured_sleeps):
    """A 429 followed by success uses the 30s rate-limit backoff, not 2**attempt."""
    emb = _bare_embedder()
    emb.client.models.embed_content.side_effect = [_RateLimitError(), _FakeResponse(2)]

    out = emb.embed(["alpha", "beta"])

    assert len(out) == 2
    assert captured_sleeps == [30.0]


def test_sustained_rate_limit_waits_two_30s_windows(captured_sleeps):
    """Sustained 429s exhaust retries after exactly two 30s waits (2x 30s)."""
    emb = _bare_embedder()
    emb.client.models.embed_content.side_effect = _RateLimitError()

    with pytest.raises(EmbeddingError):
        emb.embed(["alpha"])

    assert captured_sleeps == [30.0, 30.0]


def test_non_rate_limit_error_keeps_exponential_backoff(captured_sleeps):
    """Non-429 transient errors keep the fast 2s/4s exponential backoff."""
    emb = _bare_embedder()
    emb.client.models.embed_content.side_effect = RuntimeError("connection reset")

    with pytest.raises(EmbeddingError):
        emb.embed(["alpha"])

    assert captured_sleeps == [2, 4]
