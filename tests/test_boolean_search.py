"""Boolean word search over indexed chunk text.

These tests verify:
1. AND/OR boolean logic against the chunk store
2. Whole-word matching - "heart" must not match "hearth"
3. Case-insensitive matching in both directions
4. Regex injection prevention (query terms are escaped)
5. Metadata filters composing with the text match

Tests are designed to FAIL LOUDLY if:
- Boolean logic is inverted (AND returning OR results or vice versa)
- Matching degrades to substring, which would silently widen every query
- A query term reaches the regex engine unescaped
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deep_zotero.models import Chunk
from deep_zotero.vector_store import VectorStore


def _embedder(dimensions: int = 768) -> MagicMock:
    mock = MagicMock()
    mock.dimensions = dimensions
    mock.embed = MagicMock(
        side_effect=lambda texts, **kwargs: [[0.1] * dimensions for _ in texts]
    )
    return mock


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        page_num=1 + index,
        chunk_index=index,
        char_start=index * 100,
        char_end=index * 100 + len(text),
        section="methods",
        section_confidence=1.0,
    )


# Text chosen to separate word matching from substring matching: HEART001
# contains both "Heart" and "hearth", ELEC002 contains neither.
CORPUS = {
    "HEART001": {
        "meta": {
            "title": "Heart Rate Variability",
            "authors": "Shaffer, F.",
            "year": 2017,
            "tags": "HRV; autonomic",
            "collections": "Cardiac",
        },
        "texts": [
            "Heart rate variability reflects autonomic tone.",
            "The hearth was warm and the signal was clean.",
        ],
    },
    "ELEC002": {
        "meta": {
            "title": "Electrode Impedance",
            "authors": "Chen, L.",
            "year": 2021,
            "tags": "electrode",
            "collections": "Hardware",
        },
        "texts": [
            "Electrode impedance rises with a dry signal path.",
        ],
    },
}


@pytest.fixture
def store(temp_db_path: Path) -> VectorStore:
    s = VectorStore(temp_db_path, _embedder())
    for doc_id, spec in CORPUS.items():
        s.add_chunks(
            doc_id,
            spec["meta"],
            [_chunk(i, t) for i, t in enumerate(spec["texts"])],
        )
    return s


def docs(chunks) -> set[str]:
    return {c.metadata["doc_id"] for c in chunks}


class TestBooleanAND:
    def test_all_words_present(self, store: VectorStore):
        assert docs(store.match_chunks(["heart", "rate"], "AND")) == {"HEART001"}

    def test_missing_word_excludes_document(self, store: VectorStore):
        assert store.match_chunks(["heart", "impedance"], "AND") == []

    def test_single_word(self, store: VectorStore):
        assert docs(store.match_chunks(["signal"], "AND")) == {"HEART001", "ELEC002"}

    def test_and_is_evaluated_per_chunk(self, store: VectorStore):
        """Words in different chunks of one document do not satisfy AND."""
        assert store.match_chunks(["variability", "hearth"], "AND") == []

    def test_absent_term_narrows_without_raising(self, store: VectorStore):
        assert store.match_chunks(["heart", "zzzznotaword"], "AND") == []


class TestBooleanOR:
    def test_any_word_matches(self, store: VectorStore):
        assert docs(store.match_chunks(["heart", "electrode"], "OR")) == {
            "HEART001",
            "ELEC002",
        }

    def test_one_word_missing(self, store: VectorStore):
        assert docs(store.match_chunks(["electrode", "zzzznotaword"], "OR")) == {
            "ELEC002"
        }

    def test_all_words_missing(self, store: VectorStore):
        assert store.match_chunks(["zzzznotaword", "alsomissing"], "OR") == []


class TestWordBoundaries:
    def test_substring_does_not_match(self, store: VectorStore):
        """'heart' must not match 'hearth' - the point of word matching."""
        hits = store.match_chunks(["heart"], "AND")
        assert docs(hits) == {"HEART001"}
        assert len(hits) == 1
        assert "Heart rate" in hits[0].text

    def test_the_longer_word_is_still_reachable(self, store: VectorStore):
        hits = store.match_chunks(["hearth"], "AND")
        assert len(hits) == 1
        assert "hearth" in hits[0].text

    def test_hyphenated_term_is_matched_as_written(self, store: VectorStore):
        assert store.match_chunks(["heart-rate"], "AND") == []


class TestCaseInsensitivity:
    def test_uppercase_query(self, store: VectorStore):
        assert docs(store.match_chunks(["HEART"], "AND")) == {"HEART001"}

    def test_mixed_case_query(self, store: VectorStore):
        assert docs(store.match_chunks(["HeArT", "RaTe"], "AND")) == {"HEART001"}

    def test_lowercase_query_matches_capitalised_text(self, store: VectorStore):
        assert docs(store.match_chunks(["electrode"], "AND")) == {"ELEC002"}


class TestRegexInjection:
    """Query terms are escaped, so regex metacharacters stay literal."""

    def test_wildcard_does_not_match_everything(self, store: VectorStore):
        assert store.match_chunks([".*"], "AND") == []

    def test_alternation_is_literal(self, store: VectorStore):
        assert store.match_chunks(["heart|electrode"], "AND") == []

    def test_character_class_is_literal(self, store: VectorStore):
        assert store.match_chunks(["[a-z]+"], "AND") == []

    def test_unbalanced_bracket_does_not_raise(self, store: VectorStore):
        assert store.match_chunks(["signal("], "AND") == []


class TestWordFilter:
    def test_single_term_is_bare(self):
        """Chroma rejects $and/$or with fewer than two operands."""
        assert "$regex" in VectorStore.build_word_filter(["heart"])

    def test_and_composes(self):
        f = VectorStore.build_word_filter(["heart", "rate"], "AND")
        assert list(f) == ["$and"] and len(f["$and"]) == 2

    def test_or_composes(self):
        f = VectorStore.build_word_filter(["heart", "rate"], "OR")
        assert list(f) == ["$or"] and len(f["$or"]) == 2

    def test_operator_case_insensitive(self):
        assert list(VectorStore.build_word_filter(["a", "b"], "or")) == ["$or"]

    def test_repeated_term_still_matches(self, store: VectorStore):
        """A duplicated term is a redundant clause, not an empty result."""
        assert docs(store.match_chunks(["heart", "heart"], "AND")) == {"HEART001"}


class TestFiltersAndEdgeCases:
    def test_no_words_returns_empty(self, store: VectorStore):
        assert store.match_chunks([], "AND") == []

    def test_year_filter_applies(self, store: VectorStore):
        hits = store.match_chunks(["signal"], "AND", where={"year": {"$gte": 2020}})
        assert docs(hits) == {"ELEC002"}

    def test_chunk_type_filter_applies(self, store: VectorStore):
        hits = store.match_chunks(
            ["signal"], "AND", where={"chunk_type": {"$eq": "figure"}}
        )
        assert hits == []

    def test_results_carry_text_and_page(self, store: VectorStore):
        hits = store.match_chunks(["electrode"], "AND")
        assert len(hits) == 1
        assert "Electrode" in hits[0].text
        assert hits[0].metadata["page_num"] == 1
