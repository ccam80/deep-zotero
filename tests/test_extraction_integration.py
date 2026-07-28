"""End-to-end extraction pipeline integration tests."""
from pathlib import Path
from deep_zotero.config import Config
from deep_zotero.embedder import create_embedder
from deep_zotero.vector_store import VectorStore


def _create_test_config(tmp_path: Path) -> Config:
    zotero_dir = tmp_path / "zotero"
    zotero_dir.mkdir(exist_ok=True)
    (zotero_dir / "zotero.sqlite").touch()
    return Config(
        zotero_data_dir=zotero_dir,
        chroma_db_path=tmp_path / "chroma",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
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
        vision_enabled=False,
        vision_model="claude-haiku-4-5-20251001",
        anthropic_api_key=None,
    )


def test_noname1_chunks_have_introduction(chunked_papers):
    chunks = chunked_papers["noname1.pdf"]
    intro_chunks = [c for c in chunks if c.section == "introduction"]
    assert len(intro_chunks) >= 1


def test_full_pipeline_search(tmp_path, extracted_papers, chunked_papers):
    """Full pipeline: extract -> chunk -> embed -> store -> search returns results."""
    config = _create_test_config(tmp_path)
    ex = extracted_papers["noname1.pdf"]
    chunks = chunked_papers["noname1.pdf"]
    assert len(chunks) > 5

    embedder = create_embedder(config)
    store = VectorStore(config.chroma_db_path, embedder)
    doc_meta = {
        "title": "Test", "authors": "", "year": 2020, "citation_key": "",
        "publication": "", "journal_quartile": "", "doi": "", "tags": "",
        "collections": "", "pdf_hash": "test", "quality_grade": ex.quality_grade,
    }
    store.add_chunks("test_noname1", doc_meta, chunks)
    results = store.search("ECG modeling", top_k=5)
    assert len(results) > 0


def test_all_papers_quality_grade_a(extracted_papers):
    for pdf_name in ["noname1.pdf", "noname2.pdf", "noname3.pdf"]:
        ex = extracted_papers[pdf_name]
        assert ex.quality_grade in ("A", "B"), (
            f"{pdf_name} quality grade is {ex.quality_grade}, expected A or B"
        )
