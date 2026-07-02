"""End-to-end tests against real academic papers."""
from pathlib import Path
from deep_zotero.config import Config
from deep_zotero.embedder import create_embedder
from deep_zotero.vector_store import VectorStore
from deep_zotero.retriever import Retriever


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
        stats_sample_limit=10000,
        ocr_language="eng",
        openalex_email=None,
        vision_enabled=False,
        vision_model="claude-haiku-4-5-20251001",
        anthropic_api_key=None,
    )


def test_all_papers_produce_multiple_sections(chunked_papers):
    for pdf_name in ["noname1.pdf", "noname2.pdf", "noname3.pdf"]:
        chunks = chunked_papers[pdf_name]
        sections_found = set(c.section for c in chunks)
        assert len(sections_found) >= 3, f"{pdf_name}: only {len(sections_found)} sections in chunks: {sections_found}"


def test_all_papers_produce_enough_chunks(chunked_papers):
    for pdf_name in ["noname1.pdf", "noname2.pdf", "noname3.pdf"]:
        chunks = chunked_papers[pdf_name]
        assert len(chunks) > 20, f"{pdf_name}: only {len(chunks)} chunks"


def test_full_pipeline_retriever(tmp_path, extracted_papers, chunked_papers):
    """Full pipeline: extract -> chunk -> embed -> store -> retriever.search() returns results."""
    config = _create_test_config(tmp_path)
    ex = extracted_papers["noname1.pdf"]
    chunks = chunked_papers["noname1.pdf"]

    embedder = create_embedder(config)
    store = VectorStore(config.chroma_db_path, embedder)
    doc_meta = {
        "title": "Test Paper", "authors": "Test Author", "year": 2020,
        "citation_key": "test2020", "publication": "Test Journal",
        "journal_quartile": "", "doi": "", "tags": "", "collections": "",
        "pdf_hash": "test", "quality_grade": ex.quality_grade,
    }
    store.add_chunks("test_noname1", doc_meta, chunks)

    retriever = Retriever(store)
    results = retriever.search("ECG modeling", top_k=5)
    assert len(results) > 0
