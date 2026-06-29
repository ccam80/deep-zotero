"""Unit tests for pdf_processor module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_zotero.feature_extraction.vision_extract import AgentResponse


def test_layout_import_order():
    """pymupdf.layout must be importable."""
    import pymupdf.layout


def test_noname1_page_count(extracted_papers):
    ex = extracted_papers["noname1.pdf"]
    assert len(ex.pages) == 19


def test_noname1_quality(extracted_papers):
    ex = extracted_papers["noname1.pdf"]
    assert ex.quality_grade == "A"
    assert ex.stats["empty_pages"] == 0
    assert len(ex.full_markdown) > 50000


def test_noname2_page_count(extracted_papers):
    ex = extracted_papers["noname2.pdf"]
    assert len(ex.pages) == 21


def test_noname2_quality(extracted_papers):
    ex = extracted_papers["noname2.pdf"]
    assert ex.quality_grade == "A"
    assert ex.stats["empty_pages"] == 0
    assert len(ex.full_markdown) > 50000


def test_noname3_page_count(extracted_papers):
    ex = extracted_papers["noname3.pdf"]
    assert len(ex.pages) == 14


def test_noname3_quality(extracted_papers):
    ex = extracted_papers["noname3.pdf"]
    assert ex.quality_grade == "A"
    assert ex.stats["empty_pages"] == 0
    assert len(ex.full_markdown) > 50000


# ---------------------------------------------------------------------------
# Helpers for vision extraction tests
# ---------------------------------------------------------------------------

def _make_agent_response(
    *,
    headers=None,
    rows=None,
    caption="",
    table_label=None,
    footnotes="",
    parse_success=True,
    is_incomplete=False,
    incomplete_reason="",
    recrop_needed=False,
    recrop_bbox_pct=None,
    raw_response="{}",
) -> AgentResponse:
    return AgentResponse(
        headers=headers or [],
        rows=rows or [],
        footnotes=footnotes,
        table_label=table_label,
        caption=caption,
        is_incomplete=is_incomplete,
        incomplete_reason=incomplete_reason,
        raw_shape=(len(rows or []), len(headers or [])),
        parse_success=parse_success,
        raw_response=raw_response,
        recrop_needed=recrop_needed,
        recrop_bbox_pct=recrop_bbox_pct,
    )


def _make_detected_caption(text="Table 1", caption_type="table", bbox=None, y_center=100.0):
    """Build a DetectedCaption instance with the correct fields."""
    from deep_zotero.feature_extraction.captions import DetectedCaption
    return DetectedCaption(
        text=text,
        bbox=bbox or (0.0, 50.0, 400.0, 70.0),
        y_center=y_center,
        caption_type=caption_type,
        number=None,
    )


def _make_page_mock(page_text="some text"):
    """Build a pymupdf Page mock."""
    page = MagicMock()
    page.rect = MagicMock()
    page.rect.x0 = 0.0
    page.rect.x1 = 400.0
    page.rect.y1 = 800.0
    page.get_text.return_value = page_text
    return page


def _make_doc_mock(pages=None):
    """Build a pymupdf Document mock with given pages."""
    doc = MagicMock()
    doc.__getitem__ = MagicMock(side_effect=lambda i: (pages or [])[i])
    doc.close = MagicMock()
    return doc


def _make_mock_api(responses):
    """Build a VisionAPI mock whose extract_tables_batch returns responses."""
    api = MagicMock()
    api.extract_tables_batch.return_value = responses
    return api


def _extract_and_resolve(pdf, vision_api):
    """Call extract_document + resolve_pending_vision (must be called under mocks)."""
    from deep_zotero.pdf_processor import extract_document, resolve_pending_vision
    extraction = extract_document(pdf, vision_api=vision_api)
    if extraction.pending_vision is not None:
        resolve_pending_vision({"test": extraction}, vision_api)
    return extraction


def _make_page_chunk(page_num=1):
    return {
        "metadata": {"page_number": page_num},
        "text": "",
        "page_boxes": [],
        "toc_items": [],
    }


# ---------------------------------------------------------------------------
# Shared patch targets
# ---------------------------------------------------------------------------

_PYMUPDF4LLM = "deep_zotero.pdf_processor.pymupdf4llm"
_PYMUPDF_OPEN = "deep_zotero.pdf_processor.pymupdf.open"
_FIND_ALL_CAPTIONS = "deep_zotero.pdf_processor.find_all_captions"
_DETECT_SECTIONS = "deep_zotero.pdf_processor._detect_sections"
_DETECT_ABSTRACT = "deep_zotero.pdf_processor._detect_abstract"
_COMPUTE_STATS = "deep_zotero.pdf_processor._compute_stats"
_COMPUTE_COMPLETENESS = "deep_zotero.pdf_processor._compute_completeness"
_ASSIGN_HEADING = "deep_zotero.pdf_processor._assign_heading_captions"
_ASSIGN_CONTINUATION = "deep_zotero.pdf_processor._assign_continuation_captions"
_EXTRACT_FIGURES = "deep_zotero.pdf_processor._extract_figures_for_page"


def _make_completeness_mock(grade="A"):
    comp = MagicMock()
    comp.grade = grade
    return comp


def _base_patches(page, extra_captions=None):
    """Return a dict of patches for a minimal extract_document call."""
    captions = extra_captions if extra_captions is not None else []
    return {
        _PYMUPDF4LLM: MagicMock(to_markdown=MagicMock(return_value=[_make_page_chunk(1)])),
        _PYMUPDF_OPEN: MagicMock(return_value=_make_doc_mock([page])),
        _FIND_ALL_CAPTIONS: MagicMock(return_value=captions),
        _DETECT_SECTIONS: MagicMock(return_value=[]),
        _DETECT_ABSTRACT: MagicMock(return_value=None),
        _COMPUTE_STATS: MagicMock(return_value={"empty_pages": 0}),
        _COMPUTE_COMPLETENESS: MagicMock(return_value=_make_completeness_mock()),
        _ASSIGN_HEADING: MagicMock(),
        _ASSIGN_CONTINUATION: MagicMock(),
        _EXTRACT_FIGURES: MagicMock(return_value=[]),
    }


# ---------------------------------------------------------------------------
# TestExtractDocument
# ---------------------------------------------------------------------------

class TestExtractDocument:

    def test_vision_api_none_returns_empty_tables(self, tmp_path):
        """When vision_api is None, tables == [] and vision_details is None."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()
        patches = _base_patches(page)

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = extract_document(pdf, vision_api=None)

        assert extraction.tables == []
        assert extraction.vision_details is None

    def test_vision_api_populates_tables(self, tmp_path):
        """With vision_api, tables are populated from vision responses."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap1 = _make_detected_caption("Table 1", y_center=100.0)
        cap2 = _make_detected_caption("Table 2", bbox=(0.0, 300.0, 400.0, 320.0), y_center=310.0)

        resp1 = _make_agent_response(headers=["A", "B"], rows=[["1", "2"]], caption="Table 1")
        resp2 = _make_agent_response(headers=["X"], rows=[["val"]], caption="Table 2")
        mock_api = _make_mock_api([resp1, resp2])

        patches = _base_patches(page, extra_captions=[cap1, cap2])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert len(extraction.tables) == 2
        assert extraction.tables[0].extraction_strategy == "vision"
        assert extraction.tables[1].extraction_strategy == "vision"
        assert extraction.tables[0].headers == ["A", "B"]
        assert extraction.tables[1].headers == ["X"]

    def test_vision_caption_used(self, tmp_path):
        """Vision caption is used for ExtractedTable.caption."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        resp = _make_agent_response(
            headers=["Col"],
            rows=[["val"]],
            caption="Table 1. Full demographics",
        )
        mock_api = _make_mock_api([resp])
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert extraction.tables[0].caption == "Table 1. Full demographics"

    def test_vision_caption_fallback_to_text_layer(self, tmp_path):
        """When vision caption is empty, text-layer caption is used."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        resp = _make_agent_response(headers=["Col"], rows=[["val"]], caption="")
        mock_api = _make_mock_api([resp])
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert extraction.tables[0].caption == "Table 1"

    def test_recrop_replaces_response(self, tmp_path):
        """When recrop response is not incomplete, it replaces the original."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        original_resp = _make_agent_response(
            headers=["ColA"],
            rows=[["old"]],
            caption="Table 1",
            recrop_needed=True,
            recrop_bbox_pct=[10.0, 10.0, 90.0, 90.0],
        )
        recrop_resp = _make_agent_response(
            headers=["ColA", "ColB"],
            rows=[["new", "data"]],
            caption="Table 1. Better",
            is_incomplete=False,
        )
        mock_api = MagicMock()
        mock_api.extract_tables_batch.side_effect = [
            [original_resp],
            [recrop_resp],
        ]
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert len(extraction.tables) == 1
        assert extraction.tables[0].headers == ["ColA", "ColB"]
        assert extraction.tables[0].rows == [["new", "data"]]

    def test_incomplete_recrop_falls_back_to_fullpage(self, tmp_path):
        """3-wave design: an incomplete recrop triggers a full-page retry (wave 3);
        when the full-page result has content, it replaces the incomplete original.
        """
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        original_resp = _make_agent_response(
            headers=["ColA"],
            rows=[["original"]],
            caption="Table 1",
            recrop_needed=True,
            recrop_bbox_pct=[10.0, 10.0, 90.0, 90.0],
        )
        recrop_resp = _make_agent_response(
            headers=["ColA"],
            rows=[["partial"]],
            caption="Table 1",
            is_incomplete=True,
            incomplete_reason="bottom edge cut off",
        )
        fullpage_resp = _make_agent_response(
            headers=["ColA"],
            rows=[["fullpage"]],
            caption="Table 1",
        )
        mock_api = MagicMock()
        # Wave 1 (initial) -> recrop requested; Wave 2 (recrop) -> incomplete;
        # Wave 3 (full-page follow-up) -> good content.
        mock_api.extract_tables_batch.side_effect = [
            [original_resp],
            [recrop_resp],
            [fullpage_resp],
        ]
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert mock_api.extract_tables_batch.call_count == 3
        assert len(extraction.tables) == 1
        assert extraction.tables[0].rows == [["fullpage"]]

    def test_incomplete_recrop_keeps_original_when_fullpage_fails(self, tmp_path):
        """3-wave design: when the wave-3 full-page retry also fails to parse,
        the valid-but-incomplete original response is kept.
        """
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        original_resp = _make_agent_response(
            headers=["ColA"],
            rows=[["original"]],
            caption="Table 1",
            recrop_needed=True,
            recrop_bbox_pct=[10.0, 10.0, 90.0, 90.0],
        )
        recrop_resp = _make_agent_response(
            headers=["ColA"],
            rows=[["partial"]],
            caption="Table 1",
            is_incomplete=True,
            incomplete_reason="bottom edge cut off",
        )
        fullpage_resp = _make_agent_response(parse_success=False)
        mock_api = MagicMock()
        mock_api.extract_tables_batch.side_effect = [
            [original_resp],
            [recrop_resp],
            [fullpage_resp],
        ]
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert mock_api.extract_tables_batch.call_count == 3
        assert len(extraction.tables) == 1
        assert extraction.tables[0].rows == [["original"]]

    def test_failed_parse_skipped(self, tmp_path):
        """Responses with parse_success=False are not converted to ExtractedTable."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        resp = _make_agent_response(parse_success=False)
        mock_api = _make_mock_api([resp])
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert len(extraction.tables) == 0

    def test_vision_details_populated(self, tmp_path):
        """vision_details is populated with one entry per table crop."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        resp = _make_agent_response(
            headers=["Col"],
            rows=[["val"]],
            caption="Table 1. Full title",
        )
        mock_api = _make_mock_api([resp])
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert extraction.vision_details is not None
        assert len(extraction.vision_details) == 1
        detail = extraction.vision_details[0]
        assert "text_layer_caption" in detail
        assert "vision_caption" in detail
        assert "recropped" in detail
        assert "parse_success" in detail
        assert detail["parse_success"] is True
        assert isinstance(detail["text_layer_caption"], str)
        assert isinstance(detail["recropped"], bool)

    def test_cell_cleaning_applied(self, tmp_path):
        """Ligatures and leading zeros in vision output are cleaned."""
        from deep_zotero.pdf_processor import extract_document

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        page = _make_page_mock()

        cap = _make_detected_caption("Table 1")
        # \ufb00 = "ff" ligature; "e\ufb00ect" -> "effect"
        resp = _make_agent_response(
            headers=["e\ufb00ect"],
            rows=[[".047"]],
            caption="Table 1",
        )
        mock_api = _make_mock_api([resp])
        patches = _base_patches(page, extra_captions=[cap])

        with (
            patch(_PYMUPDF4LLM + ".to_markdown", patches[_PYMUPDF4LLM].to_markdown),
            patch(_PYMUPDF_OPEN, patches[_PYMUPDF_OPEN]),
            patch(_FIND_ALL_CAPTIONS, patches[_FIND_ALL_CAPTIONS]),
            patch(_DETECT_SECTIONS, patches[_DETECT_SECTIONS]),
            patch(_DETECT_ABSTRACT, patches[_DETECT_ABSTRACT]),
            patch(_COMPUTE_STATS, patches[_COMPUTE_STATS]),
            patch(_COMPUTE_COMPLETENESS, patches[_COMPUTE_COMPLETENESS]),
            patch(_ASSIGN_HEADING, patches[_ASSIGN_HEADING]),
            patch(_ASSIGN_CONTINUATION, patches[_ASSIGN_CONTINUATION]),
            patch(_EXTRACT_FIGURES, patches[_EXTRACT_FIGURES]),
        ):
            extraction = _extract_and_resolve(pdf, mock_api)

        assert len(extraction.tables) == 1
        assert extraction.tables[0].headers == ["effect"]
        assert extraction.tables[0].rows[0][0] == "0.047"
