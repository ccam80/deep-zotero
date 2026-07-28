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


# ---------------------------------------------------------------------------
# Backoff policy — which wait for which failure
# ---------------------------------------------------------------------------

def test_rate_limited_uses_fixed_quota_window():
    """429s wait the full configured window, not a growing exponential."""
    emb = _bare_embedder()

    assert emb._backoff_seconds(rate_limited=True, attempt=1) == 30.0
    assert emb._backoff_seconds(rate_limited=True, attempt=2) == 30.0


def test_transient_error_uses_exponential_backoff():
    """Non-429 failures keep the fast 2s/4s exponential backoff."""
    emb = _bare_embedder()

    assert emb._backoff_seconds(rate_limited=False, attempt=1) == 2
    assert emb._backoff_seconds(rate_limited=False, attempt=2) == 4


def test_rate_limit_backoff_is_configurable():
    """The quota window comes from config, not a literal."""
    emb = _bare_embedder(rate_limit_backoff=45.0)

    assert emb._backoff_seconds(rate_limited=True, attempt=1) == 45.0


@pytest.mark.parametrize("exc", [
    _RateLimitError(),
    Exception("429 Too Many Requests"),
    Exception("RESOURCE_EXHAUSTED: quota exceeded"),
])
def test_rate_limit_detected(exc):
    assert Embedder._is_rate_limit(exc) is True


@pytest.mark.parametrize("exc", [
    RuntimeError("connection reset"),
    Exception("500 Internal Server Error"),
])
def test_non_rate_limit_not_detected(exc):
    assert Embedder._is_rate_limit(exc) is False


# ---------------------------------------------------------------------------
# Retry loop — backoff set to 0 so the loop runs at full speed
# ---------------------------------------------------------------------------

def test_recovers_after_rate_limit():
    """A 429 followed by success returns the embeddings."""
    emb = _bare_embedder(rate_limit_backoff=0)
    emb.client.models.embed_content.side_effect = [_RateLimitError(), _FakeResponse(2)]

    out = emb.embed(["alpha", "beta"])

    assert len(out) == 2
    assert emb.client.models.embed_content.call_count == 2


def test_sustained_rate_limit_exhausts_retries():
    """Sustained 429s raise after exactly max_retries attempts."""
    emb = _bare_embedder(rate_limit_backoff=0)
    emb.client.models.embed_content.side_effect = _RateLimitError()

    with pytest.raises(EmbeddingError):
        emb.embed(["alpha"])

    assert emb.client.models.embed_content.call_count == 3


def test_sustained_transient_error_exhausts_retries():
    """Non-429 failures also stop at max_retries. Single attempt: never sleeps."""
    emb = _bare_embedder(max_retries=1)
    emb.client.models.embed_content.side_effect = RuntimeError("connection reset")

    with pytest.raises(EmbeddingError):
        emb.embed(["alpha"])

    assert emb.client.models.embed_content.call_count == 1
