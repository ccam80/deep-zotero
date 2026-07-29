"""Tests for composite reranking."""
import pytest
from deep_zotero.models import RetrievalResult
from deep_zotero.reranker import (
    Reranker,
    validate_section_weights,
    validate_journal_weights,
    DEFAULT_SECTION_WEIGHTS,
    VALID_SECTIONS,
    VALID_QUARTILES,
)


def make_result(
    score: float = 0.8,
    section: str = "results",
    section_confidence: float = 1.0,
    journal_quartile: str | None = "Q1",
    **kwargs
) -> RetrievalResult:
    """Helper to create test RetrievalResult."""
    defaults = {
        "chunk_id": "test_chunk",
        "text": "Test text",
        "doc_id": "test_doc",
        "doc_title": "Test Title",
        "authors": "Test Author",
        "year": 2024,
        "page_num": 1,
        "chunk_index": 0,
    }
    defaults.update(kwargs)
    return RetrievalResult(
        score=score,
        section=section,
        section_confidence=section_confidence,
        journal_quartile=journal_quartile,
        **defaults
    )


class TestReranker:
    """Test the Reranker class."""

    def test_empty_results(self):
        reranker = Reranker()
        assert reranker.rerank([]) == []

    def test_single_result(self):
        reranker = Reranker()
        result = make_result(score=0.9, section="results", journal_quartile="Q1")
        reranked = reranker.rerank([result])
        assert len(reranked) == 1
        assert reranked[0].chunk_id == result.chunk_id
        assert reranked[0].composite_score is not None

    def test_composite_score_populated(self):
        """Reranking should populate composite_score on results."""
        reranker = Reranker()
        result = make_result(score=0.8, section="results", journal_quartile="Q1")
        reranked = reranker.rerank([result])
        assert len(reranked) == 1
        # score^0.7 * 1.0 (results) * 1.0 (Q1)
        expected = 0.8 ** 0.7 * 1.0 * 1.0
        assert abs(reranked[0].composite_score - expected) < 0.001

    def test_alpha_configurable(self):
        """Alpha parameter should affect scoring."""
        result = make_result(score=0.8, section="results", journal_quartile="Q1")

        reranker_default = Reranker(alpha=0.7)
        reranker_high = Reranker(alpha=1.0)

        score_default = reranker_default.score_result(result)
        score_high = reranker_high.score_result(result)

        # alpha=1.0 should give lower score for similarity < 1.0
        assert score_high < score_default

    def test_journal_weights_override(self):
        """Custom journal weights should change ranking."""
        reranker = Reranker()

        q1_chunk = make_result(
            score=0.9,
            section="results",
            journal_quartile="Q1",
            chunk_id="q1"
        )
        q4_chunk = make_result(
            score=0.9,
            section="results",
            journal_quartile="Q4",
            chunk_id="q4"
        )

        # Default: Q1 (1.0) > Q4 (0.45)
        reranked = reranker.rerank([q4_chunk, q1_chunk])
        assert reranked[0].chunk_id == "q1"

        # Override: boost Q4 to 1.0
        reranked = reranker.rerank(
            [q4_chunk, q1_chunk],
            journal_weights={"Q4": 1.0}
        )
        # Now they have equal weights
        assert len(reranked) == 2

    def test_unknown_journal_weight(self):
        """The 'unknown' key should map to None quartile."""
        reranker = Reranker()

        result_none = make_result(journal_quartile=None, chunk_id="none")
        result_q1 = make_result(journal_quartile="Q1", chunk_id="q1")

        # Override: make unknown journals top tier
        reranked = reranker.rerank(
            [result_none, result_q1],
            journal_weights={"unknown": 1.0, "Q1": 0.5}
        )
        assert reranked[0].chunk_id == "none"

    def test_reranking_changes_order(self):
        """Results section should beat introduction even with lower similarity."""
        reranker = Reranker()

        # Higher similarity but introduction section
        intro = make_result(
            score=0.95,
            section="introduction",
            journal_quartile="Q1",
            chunk_id="intro"
        )
        # Lower similarity but results section
        results = make_result(
            score=0.85,
            section="results",
            journal_quartile="Q1",
            chunk_id="results"
        )

        reranked = reranker.rerank([intro, results])

        # Results should come first due to higher section weight
        assert reranked[0].chunk_id == "results"
        assert reranked[1].chunk_id == "intro"

    def test_journal_quartile_affects_ranking(self):
        """Q1 journal should beat Q4 journal with same similarity and section."""
        reranker = Reranker()

        q4_result = make_result(
            score=0.9,
            section="results",
            journal_quartile="Q4",
            chunk_id="q4"
        )
        q1_result = make_result(
            score=0.9,
            section="results",
            journal_quartile="Q1",
            chunk_id="q1"
        )

        reranked = reranker.rerank([q4_result, q1_result])

        assert reranked[0].chunk_id == "q1"
        assert reranked[1].chunk_id == "q4"

    def test_section_weight_override(self):
        """Custom section weights should change ranking."""
        reranker = Reranker()

        results_chunk = make_result(
            score=0.9,
            section="results",
            journal_quartile="Q1",
            chunk_id="results"
        )
        discussion_chunk = make_result(
            score=0.9,
            section="discussion",
            journal_quartile="Q1",
            chunk_id="discussion"
        )

        # Default: results (1.0) > discussion (0.65)
        reranked = reranker.rerank([discussion_chunk, results_chunk])
        assert reranked[0].chunk_id == "results"

        # Override: boost discussion to 1.0, keep results at default
        reranked = reranker.rerank(
            [discussion_chunk, results_chunk],
            section_weights={"discussion": 1.0}
        )
        # Now they have equal section weight, should maintain order by similarity
        # (both 0.9, so order depends on input order after stable sort)
        assert len(reranked) == 2

    def test_section_weight_zero_excludes(self):
        """Setting section weight to 0 should exclude those results."""
        reranker = Reranker()

        results_chunk = make_result(section="results", chunk_id="results")
        intro_chunk = make_result(section="introduction", chunk_id="intro")

        reranked = reranker.rerank(
            [results_chunk, intro_chunk],
            section_weights={"introduction": 0}
        )

        assert len(reranked) == 1
        assert reranked[0].chunk_id == "results"

    def test_weight_clamping(self):
        """Weights outside 0-1 should be clamped."""
        reranker = Reranker()

        result = make_result(section="results")

        # Weight > 1 should be clamped to 1
        reranked = reranker.rerank([result], section_weights={"results": 2.0})
        assert len(reranked) == 1

        # Negative weight should be clamped to 0 (excluded)
        reranked = reranker.rerank([result], section_weights={"results": -0.5})
        assert len(reranked) == 0

    def test_unknown_section_uses_default(self):
        """Unknown sections should use default weight of 0.7."""
        reranker = Reranker()

        unknown = make_result(section="unknown", chunk_id="unknown")
        intro = make_result(section="introduction", chunk_id="intro")

        # unknown (0.7) > introduction (0.5)
        reranked = reranker.rerank([intro, unknown])
        assert reranked[0].chunk_id == "unknown"

    def test_none_quartile_uses_default(self):
        """None/empty quartile should use default weight of 0.7."""
        reranker = Reranker()

        result_none = make_result(journal_quartile=None)
        result_empty = make_result(journal_quartile="")

        # Both should work and produce scores
        reranked = reranker.rerank([result_none, result_empty])
        assert len(reranked) == 2

    def test_score_result(self):
        """Test scoring a single result."""
        reranker = Reranker()

        result = make_result(score=0.8, section="results", journal_quartile="Q1")
        score = reranker.score_result(result)

        # score^0.7 * 1.0 (results) * 1.0 (Q1)
        expected = 0.8 ** 0.7 * 1.0 * 1.0
        assert abs(score - expected) < 0.001


class TestValidateSectionWeights:
    """Test section_weights validation."""

    def test_valid_weights(self):
        errors = validate_section_weights({"results": 1.0, "methods": 0.5})
        assert errors == []

    def test_empty_dict_valid(self):
        errors = validate_section_weights({})
        assert errors == []

    def test_invalid_section_name(self):
        errors = validate_section_weights({"invalid_section": 1.0})
        assert len(errors) == 1
        assert "Unknown section" in errors[0]

    def test_invalid_value_type(self):
        errors = validate_section_weights({"results": "high"})
        assert len(errors) == 1
        assert "must be numeric" in errors[0]

    def test_invalid_key_type(self):
        errors = validate_section_weights({123: 1.0})
        assert len(errors) == 1
        assert "must be string" in errors[0]

    def test_not_a_dict(self):
        errors = validate_section_weights("not a dict")
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]


class TestValidateJournalWeights:
    """Test journal_weights validation."""

    def test_valid_weights(self):
        errors = validate_journal_weights({"Q1": 1.0, "Q4": 0.5, "unknown": 0.7})
        assert errors == []

    def test_empty_dict_valid(self):
        errors = validate_journal_weights({})
        assert errors == []

    def test_invalid_quartile_name(self):
        errors = validate_journal_weights({"Q5": 1.0})
        assert len(errors) == 1
        assert "Unknown quartile" in errors[0]

    def test_invalid_value_type(self):
        errors = validate_journal_weights({"Q1": "high"})
        assert len(errors) == 1
        assert "must be numeric" in errors[0]

    def test_not_a_dict(self):
        errors = validate_journal_weights("not a dict")
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]

    def test_valid_quartiles_constant(self):
        expected = {"Q1", "Q2", "Q3", "Q4", "unknown"}
        assert VALID_QUARTILES == expected


class TestDefaultWeights:
    """Test that default weights are correctly defined."""

    def test_all_sections_have_weights(self):
        expected_sections = {
            "results", "conclusion", "table", "figure", "methods", "abstract",
            "background", "unknown", "discussion", "introduction", "preamble",
            "appendix", "references"
        }
        assert set(DEFAULT_SECTION_WEIGHTS.keys()) == expected_sections

    def test_valid_sections_matches_defaults(self):
        assert VALID_SECTIONS == set(DEFAULT_SECTION_WEIGHTS.keys())

    def test_every_indexed_section_value_has_a_weight(self):
        """A section the indexer writes but the reranker lacks silently falls
        through to the 0.7 default and cannot be overridden, because
        validation rejects any key absent from VALID_SECTIONS. That is how
        every figure came to rank below unclassified body text."""
        from deep_zotero.server import VALID_CHUNK_TYPES

        # add_tables/add_figures write the chunk type as the section label
        placeholders = VALID_CHUNK_TYPES - {"text"}
        missing = placeholders - set(DEFAULT_SECTION_WEIGHTS)
        assert not missing, f"chunk-type sections with no weight: {missing}"
        for name in placeholders:
            assert not validate_section_weights({name: 1.0}), (
                f"{name!r} is not overridable"
            )

    def test_chunk_type_placeholders_are_ranking_neutral(self):
        """'table' and 'figure' name a chunk type, not a section: indexing
        discards the real section, so the placeholder carries no evidence
        about relevance and must not act as a penalty."""
        from deep_zotero.server import VALID_CHUNK_TYPES

        for name in VALID_CHUNK_TYPES - {"text"}:
            assert DEFAULT_SECTION_WEIGHTS[name] == 1.0, (
                f"{name!r} weight {DEFAULT_SECTION_WEIGHTS[name]} penalises "
                f"content for a section label the pipeline chose"
            )

    def test_weights_in_range(self):
        for section, weight in DEFAULT_SECTION_WEIGHTS.items():
            assert 0 <= weight <= 1, f"{section} weight {weight} out of range"
