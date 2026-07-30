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
    mock.embed_query = MagicMock(
        side_effect=lambda text, **kwargs: [0.1] * dimensions
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


class TestPaging:
    """Chroma cannot hydrate metadata for a large result set in one get()."""

    @pytest.fixture
    def many(self, temp_db_path: Path) -> VectorStore:
        s = VectorStore(temp_db_path, _embedder())
        s.add_chunks(
            "BULK001",
            {"title": "Bulk", "authors": "A", "year": 2020},
            [_chunk(i, f"chunk {i} mentions telemetry") for i in range(23)],
        )
        return s

    def test_every_page_is_returned(self, many: VectorStore):
        hits = many.match_chunks(["telemetry"], "AND", batch_size=5)
        assert len(hits) == 23

    def test_paging_does_not_duplicate(self, many: VectorStore):
        hits = many.match_chunks(["telemetry"], "AND", batch_size=5)
        assert len({h.id for h in hits}) == 23

    def test_exact_multiple_of_batch_size_terminates(self, many: VectorStore):
        """A final full page must be followed by an empty one, not an infinite loop."""
        hits = many.match_chunks(["telemetry"], "AND", batch_size=23)
        assert len(hits) == 23

    def test_unpaged_default_matches_paged(self, many: VectorStore):
        assert len(many.match_chunks(["telemetry"], "AND")) == len(
            many.match_chunks(["telemetry"], "AND", batch_size=2)
        )

    def test_context_expanded_only_for_returned_hits(self, many, mock_config, monkeypatch):
        """Expansion costs a query per chunk, so it must follow truncation."""
        from deep_zotero import server as srv
        from deep_zotero.reranker import Reranker
        from deep_zotero.retriever import Retriever

        calls = []
        real = many.get_adjacent_chunks

        def counting(*a, **k):
            calls.append(a)
            return real(*a, **k)

        monkeypatch.setattr(many, "get_adjacent_chunks", counting)
        monkeypatch.setattr(srv, "_config", mock_config)
        monkeypatch.setattr(srv, "_store", many)
        monkeypatch.setattr(srv, "_retriever", Retriever(many))
        monkeypatch.setattr(srv, "_reranker", Reranker(alpha=mock_config.rerank_alpha))

        results = srv.search_papers(required_terms=["telemetry"], top_k=3,
                                   context_chunks=1)
        assert len(results) == 3
        assert len(calls) <= 3, f"expanded {len(calls)} of 23 matches"


@pytest.fixture
def wired(store: VectorStore, mock_config, monkeypatch):
    """search_papers wired to the fixture store, bypassing lazy init."""
    from deep_zotero import server as srv
    from deep_zotero.reranker import Reranker
    from deep_zotero.retriever import Retriever

    monkeypatch.setattr(srv, "_config", mock_config)
    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "_retriever", Retriever(store))
    monkeypatch.setattr(srv, "_reranker", Reranker(alpha=mock_config.rerank_alpha))
    return srv


class TestSearchPapersLexical:
    """required_terms drives exact word matching on search_papers."""

    def test_requires_query_or_terms(self, wired):
        from deep_zotero.server import ToolError

        with pytest.raises(ToolError, match="query, required_terms"):
            wired.search_papers()

    def test_rejects_bad_operator(self, wired):
        from deep_zotero.server import ToolError

        with pytest.raises(ToolError, match="terms_operator"):
            wired.search_papers(required_terms=["heart"], terms_operator="XOR")

    def test_terms_only_needs_no_embedding(self, wired, store: VectorStore):
        results = wired.search_papers(required_terms=["heart"])
        assert {r["doc_id"] for r in results} == {"HEART001"}
        store.embedder.embed_query.assert_not_called()

    def test_terms_only_is_whole_word(self, wired):
        results = wired.search_papers(required_terms=["heart"])
        assert len(results) == 1
        assert "Heart rate" in results[0]["passage"]

    def test_terms_operator_or(self, wired):
        results = wired.search_papers(
            required_terms=["heart", "electrode"], terms_operator="OR"
        )
        assert {r["doc_id"] for r in results} == {"HEART001", "ELEC002"}

    def test_terms_operator_and_excludes(self, wired):
        assert wired.search_papers(
            required_terms=["heart", "electrode"], terms_operator="AND"
        ) == []

    def test_query_with_terms_constrains_the_search(self, wired, store: VectorStore):
        """The term is a hard constraint, not a filter on what ranked well."""
        results = wired.search_papers(query="cardiac autonomic control",
                                      required_terms=["electrode"])
        assert {r["doc_id"] for r in results} == {"ELEC002"}
        store.embedder.embed_query.assert_called()

    def test_terms_respect_metadata_filters(self, wired):
        assert wired.search_papers(required_terms=["signal"], year_min=2020) != []
        assert wired.search_papers(required_terms=["signal"], year_max=2000) == []


class TestExactMatchFilters:
    """sections and journal_quartiles exclude, and work without a query."""

    @staticmethod
    def _add(store: VectorStore, doc, section, quartile, text, year=1993):
        store.add_chunks(
            doc,
            {"title": f"Paper {doc}", "authors": "Olufsen, M.", "year": year,
             "journal_quartile": quartile, "publication": "J Physiol"},
            [Chunk(text=text, page_num=1, chunk_index=0, char_start=0,
                   char_end=len(text), section=section, section_confidence=1.0)],
        )

    @pytest.fixture
    def corpus(self, store: VectorStore):
        self._add(store, "Q1RES", "results", "Q1", "baroreflex gain rose")
        self._add(store, "Q4RES", "results", "Q4", "baroreflex gain rose")
        self._add(store, "Q1INT", "introduction", "Q1", "baroreflex gain rose")
        self._add(store, "NORES", "results", "", "baroreflex gain rose")
        return store

    def test_sections_excludes_other_sections(self, corpus, wired):
        results = wired.search_papers(required_terms=["baroreflex"], sections=["results"])
        assert {r["doc_id"] for r in results} == {"Q1RES", "Q4RES", "NORES"}

    def test_quartiles_excludes_other_quartiles(self, corpus, wired):
        results = wired.search_papers(
            required_terms=["baroreflex"], journal_quartiles=["Q1"]
        )
        assert {r["doc_id"] for r in results} == {"Q1RES", "Q1INT"}

    def test_unknown_quartile_selects_unranked(self, corpus, wired):
        results = wired.search_papers(
            required_terms=["baroreflex"], journal_quartiles=["unknown"]
        )
        assert {r["doc_id"] for r in results} == {"NORES"}

    def test_filters_combine(self, corpus, wired):
        results = wired.search_papers(
            required_terms=["baroreflex"],
            sections=["results"],
            journal_quartiles=["Q1"],
        )
        assert {r["doc_id"] for r in results} == {"Q1RES"}

    def test_filters_apply_without_a_query(self, corpus, wired):
        """Weights cannot filter here: the lexical path bypasses reranking."""
        by_weight = wired.search_papers(
            required_terms=["baroreflex"],
            journal_weights={"Q2": 0.0, "Q3": 0.0, "Q4": 0.0, "unknown": 0.0},
        )
        assert "Q4RES" in {r["doc_id"] for r in by_weight}

        by_filter = wired.search_papers(
            required_terms=["baroreflex"], journal_quartiles=["Q1"]
        )
        assert "Q4RES" not in {r["doc_id"] for r in by_filter}

    def test_filters_apply_with_a_query(self, corpus, wired):
        results = wired.search_papers(
            query="baroreflex gain", sections=["results"], journal_quartiles=["Q1"]
        )
        assert {r["doc_id"] for r in results} == {"Q1RES"}

    def test_invalid_section_rejected(self, wired):
        from deep_zotero.server import ToolError

        with pytest.raises(ToolError, match="Invalid sections"):
            wired.search_papers(query="x", sections=["conclusions"])

    def test_invalid_quartile_rejected(self, wired):
        from deep_zotero.server import ToolError

        with pytest.raises(ToolError, match="Invalid journal_quartiles"):
            wired.search_papers(query="x", journal_quartiles=["Q5"])


class TestResultCarriesChunkType:
    """search_papers labels each result and adds table/figure fields."""

    def test_text_result_is_labelled(self, wired):
        results = wired.search_papers(required_terms=["heart"])
        assert results[0]["chunk_type"] == "text"

    def test_text_result_has_no_table_or_figure_fields(self, wired):
        r = wired.search_papers(required_terms=["heart"])[0]
        assert "image_path" not in r and "num_rows" not in r

    def test_figure_result_carries_image_path(self, store: VectorStore, wired):
        store.collection.add(
            ids=["FIG003_fig_0000"],
            embeddings=[[0.1] * 768],
            documents=["Figure 1. Electrode placement on the torso."],
            metadatas=[{
                "doc_id": "FIG003", "doc_title": "Placement", "authors": "Wu, K.",
                "year": 2019, "page_num": 3, "chunk_index": 0,
                "chunk_type": "figure", "section": "figure",
                "caption": "Figure 1. Electrode placement on the torso.",
                "image_path": "/figs/FIG003_p3_0.png", "figure_index": 0,
            }],
        )
        r = next(
            x for x in wired.search_papers(required_terms=["torso"])
            if x["doc_id"] == "FIG003"
        )
        assert r["chunk_type"] == "figure"
        assert r["image_path"] == "/figs/FIG003_p3_0.png"
        assert r["caption"].startswith("Figure 1.")
        assert r["figure_index"] == 0

    def test_table_result_carries_dimensions(self, store: VectorStore, wired):
        store.collection.add(
            ids=["TAB004_table_0000"],
            embeddings=[[0.1] * 768],
            documents=["| subject | impedance |\n|---|---|\n| A | 4.2 |"],
            metadatas=[{
                "doc_id": "TAB004", "doc_title": "Impedance", "authors": "Wu, K.",
                "year": 2019, "page_num": 5, "chunk_index": 0,
                "chunk_type": "table", "section": "table",
                "table_caption": "Table 2. Impedance by subject.",
                "table_index": 1, "table_num_rows": 1, "table_num_cols": 2,
            }],
        )
        r = next(
            x for x in wired.search_papers(required_terms=["impedance"])
            if x["doc_id"] == "TAB004"
        )
        assert r["chunk_type"] == "table"
        assert r["num_rows"] == 1 and r["num_cols"] == 2
        assert r["table_index"] == 1
        assert r["caption"].startswith("Table 2.")
        assert "| subject |" in r["passage"]
