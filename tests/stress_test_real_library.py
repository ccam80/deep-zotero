"""
STRESS TEST: Real-world researcher workflow simulation.

This test picks 10 diverse papers from the user's actual Zotero library,
indexes them through the full pipeline, and simulates realistic researcher
search patterns. It tests:

1. Extraction quality: text, tables, figures, sections, OCR
2. Search accuracy: can the researcher find what they're looking for?
3. Table search: can specific results tables be found by content?
4. Figure search: can specific figures be found by caption?
5. Metadata filtering: author, collection, year range
6. Context expansion: does expanded context make sense?
7. OCR pathway: can image-only PDFs still be searched?
8. Edge cases: missing DOIs, non-standard formats, empty pages

Every failure is logged. The final report is brutally honest.
"""
from __future__ import annotations


import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from deep_zotero.feature_extraction.captions import find_all_captions
from deep_zotero.feature_extraction.debug_db import (
    clear_vision_results,
    create_extended_tables,
    write_vision_agent_result,
    write_vision_run_detail,
)
# ---------------------------------------------------------------------------
# Corpus selection: 10 papers chosen for maximum diversity
# ---------------------------------------------------------------------------
# Each entry: (item_key, short_name, why_chosen, ground_truth)
# ground_truth is a dict of expected properties the test will validate

CORPUS = [
    (
        "SCPXVBLY",
        "active-inference-tutorial",
        "Tutorial paper with equations, diagrams, algorithm boxes. Tests complex layout.",
        {
            "year": 2022,
            "author_substr": "smith",
            "collection": "Active Inference",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "discussion", "conclusion"],
            "searchable_content": "active inference free energy",
            "table_search_query": "algorithm update rules",
            "figure_search_query": "generative model graphical",
        },
    ),
    (
        "XIAINRVS",
        "huang-emd-1998",
        "Foundational EMD paper from 1998. Math-heavy, older format. Tests pre-2000 PDFs.",
        {
            "year": 1998,
            "author_substr": "huang",
            "collection": "EMD",
            "expect_tables": False,
            "expect_figures": True,
            "expect_sections": ["introduction", "conclusion"],
            "searchable_content": "empirical mode decomposition Hilbert spectrum",
            "table_search_query": None,
            "figure_search_query": "intrinsic mode function",
        },
    ),
    (
        "C626CYVT",
        "hallett-tms-primer",
        "TMS primer/review. Well-structured clinical review. Tests clean section detection.",
        {
            "year": 2007,
            "author_substr": "hallett",
            "collection": "TMS",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction"],
            "searchable_content": "transcranial magnetic stimulation motor cortex",
            "table_search_query": "stimulation parameters coil",
            "figure_search_query": "magnetic field coil",
        },
    ),
    (
        "5SIZVS65",
        "laird-fick-polyps",
        "Epidemiology paper with demographic tables. Tests table extraction accuracy.",
        {
            "year": 2016,
            "author_substr": "laird",
            "collection": "PhD",
            "expect_tables": True,
            "expect_figures": False,
            "expect_sections": ["introduction", "methods", "results", "discussion"],
            "searchable_content": "colonic polyp histopathology colonoscopy",
            "table_search_query": "polyp location demographics patient",
            "figure_search_query": None,
        },
    ),
    (
        "9GKLLJH9",
        "helm-coregulation",
        "Psychology paper on RSA coregulation. Tests social science format.",
        {
            "year": 2014,
            "author_substr": "helm",
            "collection": "Coregulation",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "methods", "results", "discussion"],
            "searchable_content": "respiratory sinus arrhythmia coregulation romantic",
            "table_search_query": "correlation coefficient RSA",
            "figure_search_query": "RSA dynamics time series",
        },
    ),
    (
        "Z9X4JVZ5",
        "roland-emg-filter",
        "IEEE engineering paper on digital filtering. Tests 2-column format, circuits.",
        {
            "year": 2019,
            "author_substr": "roland",
            "collection": "Processing",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "results", "conclusion"],
            "searchable_content": "ultra-low-power digital filtering EMG",
            "table_search_query": "power consumption filter",
            "figure_search_query": "filter frequency response",
        },
    ),
    (
        "YMWV46JA",
        "friston-life",
        "Theoretical neuroscience. Dense, abstract, math-heavy. Tests unusual structure.",
        {
            "year": 2013,
            "author_substr": "friston",
            "collection": "Friston",
            "expect_tables": False,
            "expect_figures": True,
            "expect_sections": ["introduction"],
            "searchable_content": "free energy principle self-organization",
            "table_search_query": None,
            "figure_search_query": "Markov blanket",
        },
    ),
    (
        "DPYRZTFI",
        "yang-ppv-meta",
        "Systematic review/meta-analysis. Forest plots, summary tables. Tests meta-analysis format.",
        {
            "year": 2014,
            "author_substr": "yang",
            "collection": "PPV",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "methods", "results", "discussion"],
            "searchable_content": "pulse pressure variation fluid responsiveness",
            "table_search_query": "sensitivity specificity diagnostic",
            "figure_search_query": "sensitivity specificity receiver operating",
        },
    ),
    (
        "VP3NJ74M",
        "fortune-impedance",
        "Measurement study with impedance data. Tests engineering/measurement format.",
        {
            "year": 2021,
            "author_substr": "fortune",
            "collection": "",  # Check what collection
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "methods", "results"],
            "searchable_content": "electrode skin impedance imbalance frequency",
            "table_search_query": "impedance measurement electrode",
            "figure_search_query": "impedance frequency",
        },
    ),
    (
        "AQ3D94VC",
        "reyes-lf-hrv",
        "Review of LF HRV as autonomic index. Tests review paper format.",
        {
            "year": 2013,
            "author_substr": "reyes",
            "collection": "HRV",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "conclusion"],
            "searchable_content": "low frequency heart rate variability sympathetic",
            "table_search_query": "autonomic measures",
            "figure_search_query": "heart rate variability",
        },
    ),
    # ── additional 10 papers for broader coverage ─────────────────────────
    (
        "5746WSRD",
        "osterrieder-ach-kinetics-1980",
        "Very old paper (1980), ion channel biophysics, Pflügers Archiv. Tests pre-digital-era PDF.",
        {
            "year": 1980,
            "author_substr": "osterrieder",
            "collection": "",
            "expect_tables": False,
            "expect_figures": True,
            "expect_sections": ["introduction"],
            "searchable_content": "acetylcholine potassium channel sinoatrial",
            "table_search_query": None,
            "figure_search_query": "potassium current conductance",
        },
    ),
    (
        "E4WMHZH5",
        "linssen-emg-fatigue-1993",
        "EMG fatigue study from 1993, Muscle & Nerve. Tests older experimental physiology format.",
        {
            "year": 1993,
            "author_substr": "linssen",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "methods", "results", "discussion"],
            "searchable_content": "surface EMG fatigue conduction velocity median frequency",
            "table_search_query": "median frequency conduction velocity",
            "figure_search_query": "EMG power spectrum fatigue",
        },
    ),
    (
        "HISJD5UI",
        "berntson-hrv-origins-1997",
        "Classic HRV origins paper (1997), Psychophysiology. Tests older psychophysiology journal format.",
        {
            "year": 1997,
            "author_substr": "berntson",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction"],
            "searchable_content": "heart rate variability vagal sympathetic autonomic",
            "table_search_query": "autonomic heart rate",
            "figure_search_query": "vagal sympathetic frequency response",
        },
    ),
    (
        "6KB8FWFN",
        "ats-ers-respiratory-muscle-2002",
        "ATS/ERS clinical statement (2002). Very long guideline document, no named first author. Tests clinical guideline format.",
        {
            "year": 2002,
            "author_substr": "",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction"],
            "searchable_content": "respiratory muscle testing pressure diaphragm phrenic",
            "table_search_query": "respiratory pressure measurement",
            "figure_search_query": "diaphragm pressure stimulation",
        },
    ),
    (
        "QB3J4QTQ",
        "schroeder-hrv-repeatability-2004",
        "HRV repeatability study (2004), J Electrocardiology. Statistical methods, variance components. Tests clinical stats format.",
        {
            "year": 2004,
            "author_substr": "schroeder",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "methods", "results", "discussion"],
            "searchable_content": "repeatability heart rate variability measurement variance",
            "table_search_query": "variance reproducibility heart rate",
            "figure_search_query": "heart rate variability",
        },
    ),
    (
        "SWB2E2Q9",
        "raez-emg-techniques-2006",
        "EMG techniques survey (2006), Biological Procedures Online. Classification tables, signal processing overview.",
        {
            "year": 2006,
            "author_substr": "raez",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "conclusion"],
            "searchable_content": "EMG signal detection processing classification",
            "table_search_query": "EMG classification features",
            "figure_search_query": "EMG signal motor unit",
        },
    ),
    (
        "W3FYXN62",
        "daly-hodgkin-huxley-bayesian-2015",
        "Bayesian reparametrization of Hodgkin-Huxley (2015), Royal Society Open Science. Math-heavy, ABC-SMC. Tests math/stats paper.",
        {
            "year": 2015,
            "author_substr": "daly",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "methods", "results", "discussion"],
            "searchable_content": "Hodgkin Huxley Bayesian approximate posterior reparametrization",
            "table_search_query": "parameter prior posterior estimate",
            "figure_search_query": "posterior distribution action potential",
        },
    ),
    (
        "HGGB64RC",
        "shaffer-hrv-norms-2017",
        "HRV metrics and norms overview (2017), Frontiers. Many norm tables. Tests table-heavy Frontiers format.",
        {
            "year": 2017,
            "author_substr": "shaffer",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction"],
            "searchable_content": "heart rate variability SDNN RMSSD norms metrics",
            "table_search_query": "SDNN RMSSD norms short-term",
            "figure_search_query": "heart rate variability",
        },
    ),
    (
        "RE7X5G2Q",
        "charlton-wearable-ppg-2022",
        "Wearable PPG review (2022), Proceedings of the IEEE. 2-column format, extensive references. Tests IEEE proceedings.",
        {
            "year": 2022,
            "author_substr": "charlton",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "conclusion"],
            "searchable_content": "photoplethysmography wearable cardiovascular monitoring",
            "table_search_query": "PPG wearable heart rate blood pressure",
            "figure_search_query": "photoplethysmogram waveform pulse",
        },
    ),
    (
        "ETKMWLI7",
        "flett-wearable-hr-accuracy-2025",
        "Wearable HR accuracy preprint (2025). Very recent, SSRN preprint format. Tests preprint/unpublished.",
        {
            "year": 2025,
            "author_substr": "flett",
            "collection": "",
            "expect_tables": True,
            "expect_figures": True,
            "expect_sections": ["introduction", "methods", "results", "discussion"],
            "searchable_content": "wrist-worn wearable heart rate accuracy activity",
            "table_search_query": "heart rate accuracy error wearable",
            "figure_search_query": "heart rate Bland-Altman agreement",
        },
    ),
]


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Single test assertion result."""
    test_name: str
    paper: str
    passed: bool
    detail: str
    severity: str = "MAJOR"  # MAJOR or MINOR


@dataclass
class StressTestReport:
    """Aggregate test report."""
    results: list[TestResult] = field(default_factory=list)
    extraction_summaries: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    db_path: Path | None = None

    def add(self, test_name: str, paper: str, passed: bool, detail: str,
            severity: str = "MAJOR"):
        self.results.append(TestResult(test_name, paper, passed, detail, severity))

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def major_failures(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "MAJOR")

    def to_markdown(self) -> str:
        lines = []
        lines.append("# Stress Test Report: deep-zotero")
        lines.append("")
        lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Corpus**: {len(CORPUS)} papers from live Zotero library")
        lines.append("")

        # Executive summary
        lines.append("## Executive Summary")
        lines.append("")
        total = len(self.results)
        lines.append(f"- **Total tests**: {total}")
        lines.append(f"- **Passed**: {self.passed} ({100*self.passed/total:.0f}%)" if total else "- **Passed**: 0")
        lines.append(f"- **Failed**: {self.failed}")
        lines.append(f"- **Major failures**: {self.major_failures}")
        lines.append("")

        if self.major_failures > 0:
            lines.append("> **VERDICT**: This tool is NOT reliable for production research use.")
            lines.append("> A researcher depending on this tool WILL miss important results.")
        elif self.failed > 0:
            lines.append("> **VERDICT**: Mostly functional but has rough edges that will")
            lines.append("> frustrate researchers in real use.")
        else:
            lines.append("> **VERDICT**: All tests passed. Tool appears reliable for research use.")
        lines.append("")

        # Timings
        if self.timings:
            lines.append("## Performance")
            lines.append("")
            lines.append("| Operation | Time |")
            lines.append("|-----------|------|")
            for op, t in self.timings.items():
                lines.append(f"| {op} | {t:.1f}s |")
            lines.append("")

        # Extraction quality summary
        if self.extraction_summaries:
            lines.append("## Extraction Quality per Paper")
            lines.append("")
            lines.append("| Paper | Pages | Sections | Tables | Figures | Grade | Issues |")
            lines.append("|-------|-------|----------|--------|---------|-------|--------|")
            for s in self.extraction_summaries:
                issues = s.get("issues", "none")
                lines.append(
                    f"| {s['name'][:25]} | {s['pages']} | {s['sections']} | "
                    f"{s['tables']} | {s['figures']} | {s['grade']} | {issues} |"
                )
            lines.append("")

        # Failures detail
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.append("## Failures (Detailed)")
            lines.append("")
            for r in failures:
                icon = "!!!" if r.severity == "MAJOR" else "!"
                lines.append(f"### {icon} [{r.severity}] {r.test_name} — {r.paper}")
                lines.append("")
                lines.append(f"{r.detail}")
                lines.append("")

        # Passes summary
        passes = [r for r in self.results if r.passed]
        if passes:
            lines.append("## Passes")
            lines.append("")
            lines.append("| Test | Paper | Detail |")
            lines.append("|------|-------|--------|")
            for r in passes:
                detail_short = r.detail[:80].replace("|", "/")
                lines.append(f"| {r.test_name} | {r.paper} | {detail_short} |")
            lines.append("")

        # Errors
        if self.errors:
            lines.append("## Unexpected Errors")
            lines.append("")
            for e in self.errors:
                lines.append(f"- {e}")
            lines.append("")

        # OCR test section placeholder
        lines.append("## OCR Pathway Test")
        lines.append("")
        lines.append("_(See OCR test results in the test output above)_")
        lines.append("")

        # Vision extraction report (requires debug DB)
        if self.db_path is not None:
            vision_lines = _build_vision_extraction_report(self.db_path)
            if vision_lines:
                lines.append("")
                lines.extend(vision_lines)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main stress test runner
# ---------------------------------------------------------------------------

def run_stress_test():
    """Run the full stress test and return the report and extractions."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    report = StressTestReport()
    extractions: dict[str, tuple] = {}

    # Temp dir for ephemeral data (chroma index, OCR scratch).
    # Figures go to a persistent directory alongside the report/audit.
    test_dir = Path(tempfile.mkdtemp(prefix="stress_test_"))
    chroma_path = test_dir / "chroma"
    ocr_path = test_dir / "ocr_images"

    base_dir = Path(__file__).parent.parent
    figures_path = base_dir / "_stress_test_figures"

    print(f"Test directory: {test_dir}")
    print(f"Figures directory: {figures_path}")
    print(f"=" * 70)

    try:
        # ===================================================================
        # PHASE 1: Load papers from Zotero
        # ===================================================================
        print("\n[PHASE 1] Loading papers from Zotero library...")
        from deep_zotero.zotero_client import ZoteroClient
        from deep_zotero.config import Config

        config = Config.load()
        zotero = ZoteroClient(config.zotero_data_dir)
        all_items = zotero.get_all_items_with_pdfs()
        items_by_key = {i.item_key: i for i in all_items}

        corpus_items = []
        for item_key, short_name, reason, gt in CORPUS:
            item = items_by_key.get(item_key)
            if item is None:
                report.errors.append(f"Item {item_key} ({short_name}) not found in library")
                continue
            if not item.pdf_path or not item.pdf_path.exists():
                report.errors.append(f"PDF missing for {item_key} ({short_name})")
                continue
            corpus_items.append((item, short_name, gt))
            print(f"  [{short_name}] {item.title[:60]} — {item.pdf_path.name}")

        # ===================================================================
        # PHASE 2: Extract and index all papers
        # ===================================================================
        print(f"\n[PHASE 2] Extracting and indexing {len(corpus_items)} papers...")

        from deep_zotero.pdf_processor import extract_document
        from deep_zotero.chunker import Chunker
        from deep_zotero.embedder import create_embedder
        from deep_zotero.vector_store import VectorStore
        from deep_zotero.journal_ranker import JournalRanker
        from deep_zotero.reranker import Reranker
        from deep_zotero.retriever import Retriever
        import hashlib

        # Build test config with local embeddings for speed
        test_config = Config(
            zotero_data_dir=config.zotero_data_dir,
            chroma_db_path=chroma_path,
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

        embedder = create_embedder(test_config)
        store = VectorStore(chroma_path, embedder)
        chunker = Chunker(chunk_size=400, overlap=100)
        journal_ranker = JournalRanker()
        reranker = Reranker(alpha=0.7)
        retriever = Retriever(store)

        # Construct vision API
        vision_api = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            from deep_zotero.feature_extraction.vision_api import VisionAPI
            vision_cost_log = test_dir / "vision_costs.json"
            vision_api = VisionAPI(api_key=api_key, cost_log_path=vision_cost_log)
            print(f"  Vision API enabled (model: {vision_api._model})")
        else:
            print("  Vision API disabled (no ANTHROPIC_API_KEY)")

        t_start = time.perf_counter()

        # --- Phase 2a: Extract all documents (vision specs collected, deferred) ---
        # Collects vision specs but does NOT call the Batch API yet.
        doc_data: dict[str, tuple] = {}  # item_key -> (extraction, item, short_name, gt, t_extract)

        for item, short_name, gt in corpus_items:
            print(f"\n  Extracting [{short_name}]...", end=" ", flush=True)
            t0 = time.perf_counter()
            extraction = extract_document(
                item.pdf_path,
                write_images=True,
                images_dir=figures_path / item.item_key,
                vision_api=vision_api,
            )
            t_extract = time.perf_counter() - t0
            n_pending = len(extraction.pending_vision.specs) if extraction.pending_vision else 0
            print(
                f"{t_extract:.1f}s | {len(extraction.pages)}pp | "
                f"{len(extraction.figures)}fig | "
                f"{n_pending} table specs pending"
            )
            doc_data[item.item_key] = (extraction, item, short_name, gt, t_extract)

        # --- Phase 2b: Resolve vision batch (one API call for ALL papers) ---
        if vision_api:
            from deep_zotero.pdf_processor import resolve_pending_vision
            total_specs = sum(
                len(ext.pending_vision.specs)
                for ext, _, _, _, _ in doc_data.values()
                if ext.pending_vision
            )
            print(f"\n  Resolving vision batch: {total_specs} tables across {len(doc_data)} papers...")
            t_vision = time.perf_counter()
            resolve_pending_vision(
                {k: v[0] for k, v in doc_data.items()},
                vision_api,
            )
            t_vision = time.perf_counter() - t_vision
            print(f"  Vision batch resolved in {t_vision:.1f}s")

        # --- Phase 2c: Index each document (chunk, store, report) ---
        from deep_zotero._reference_matcher import match_references, get_reference_context
        from deep_zotero.pdf_processor import SYNTHETIC_CAPTION_PREFIX
        _TAB_NUM_RE = re.compile(r"(?:Table|Tab\.?)\s+(\d+)", re.IGNORECASE)
        _FIG_NUM_RE = re.compile(r"(?:Figure|Fig\.?)\s+(\d+)", re.IGNORECASE)

        for item_key, (extraction, item, short_name, gt, t_extract) in doc_data.items():
            # Chunk
            chunks = chunker.chunk(
                extraction.full_markdown,
                extraction.pages,
                extraction.sections,
            )

            # Build doc metadata
            journal_quartile = journal_ranker.lookup(item.publication)
            h = hashlib.sha256()
            with open(item.pdf_path, "rb") as f:
                h.update(f.read(65536))
            pdf_hash = h.hexdigest()

            doc_meta = {
                "title": item.title,
                "authors": item.authors,
                "year": item.year or 0,
                "citation_key": item.citation_key,
                "publication": item.publication,
                "journal_quartile": journal_quartile or "",
                "doi": item.doi,
                "tags": item.tags,
                "collections": item.collections,
                "pdf_hash": pdf_hash,
                "quality_grade": extraction.quality_grade,
            }

            # Store text chunks
            store.add_chunks(item.item_key, doc_meta, chunks)

            # Build reference map and enrich tables/figures with body-text context
            ref_map = match_references(extraction.full_markdown, chunks, extraction.tables, extraction.figures)
            for table in extraction.tables:
                if table.caption and not table.caption.startswith(SYNTHETIC_CAPTION_PREFIX):
                    m_cap = _TAB_NUM_RE.search(table.caption)
                    if m_cap:
                        ctx = get_reference_context(extraction.full_markdown, chunks, ref_map, "table", int(m_cap.group(1)))
                        table.reference_context = ctx
            for fig in extraction.figures:
                if fig.caption and not fig.caption.startswith(SYNTHETIC_CAPTION_PREFIX):
                    m_cap = _FIG_NUM_RE.search(fig.caption)
                    if m_cap:
                        ctx = get_reference_context(extraction.full_markdown, chunks, ref_map, "figure", int(m_cap.group(1)))
                        fig.reference_context = ctx

            # Store tables
            if extraction.tables:
                store.add_tables(item.item_key, doc_meta, extraction.tables, ref_map=ref_map)

            # Store figures
            if extraction.figures:
                store.add_figures(item.item_key, doc_meta, extraction.figures, ref_map=ref_map)

            extractions[item.item_key] = (extraction, chunks, item, gt, short_name)

            n_sections = len([s for s in extraction.sections if s.label != "preamble"])
            section_labels = [s.label for s in extraction.sections if s.label not in ("preamble", "unknown")]
            print(
                f"\n  Indexed [{short_name}]: {t_extract:.1f}s | {len(extraction.pages)}pp | "
                f"{len(chunks)}ch | {len(extraction.tables)}tab | "
                f"{len(extraction.figures)}fig | "
                f"sections: {section_labels} | "
                f"grade: {extraction.quality_grade}"
            )

            # Build extraction summary for report
            comp = extraction.completeness
            issues_parts = []
            if comp and comp.figures_missing > 0:
                issues_parts.append(f"{comp.figures_missing} figs missing")
            if comp and comp.tables_missing > 0:
                issues_parts.append(f"{comp.tables_missing} tabs missing")
            if comp and comp.unknown_sections > 0:
                issues_parts.append(f"{comp.unknown_sections} unknown sections")
            if not any(s.label == "abstract" for s in extraction.sections):
                issues_parts.append("no abstract detected")

            n_real_tables = sum(1 for t in extraction.tables if not t.artifact_type)
            n_artifact_tables = sum(1 for t in extraction.tables if t.artifact_type)
            table_str = str(n_real_tables)
            if n_artifact_tables:
                table_str += f" (+{n_artifact_tables} artifacts)"

            report.extraction_summaries.append({
                "name": short_name,
                "pages": len(extraction.pages),
                "sections": n_sections,
                "tables": table_str,
                "figures": len(extraction.figures),
                "grade": extraction.quality_grade,
                "issues": "; ".join(issues_parts) if issues_parts else "none",
            })

        t_total_index = time.perf_counter() - t_start
        report.timings["Total indexing"] = t_total_index
        print(f"\n  Total indexing time: {t_total_index:.1f}s")
        print(f"  Total chunks in store: {store.count()}")

        # ===================================================================
        # PHASE 3: Validate extraction quality
        # ===================================================================
        print(f"\n[PHASE 3] Validating extraction quality...")

        for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
            # --- 3a: Sections detected ---
            section_labels = set(s.label for s in extraction.sections)
            for expected_section in gt.get("expect_sections", []):
                found = expected_section in section_labels
                report.add(
                    "section-detection",
                    short_name,
                    found,
                    f"Expected section '{expected_section}' — "
                    + (f"FOUND" if found else f"MISSING. Got: {sorted(section_labels)}"),
                    severity="MAJOR" if expected_section in ("methods", "results") else "MINOR",
                )

            # --- 3b: Tables extracted ---
            real_tables = [t for t in extraction.tables if not t.artifact_type]
            artifact_tables = [t for t in extraction.tables if t.artifact_type]

            # Report artifact tables (informational — always passes)
            if artifact_tables:
                tags = [t.artifact_type for t in artifact_tables]
                report.add(
                    "artifact-tables-tagged",
                    short_name,
                    True,
                    f"{len(artifact_tables)} layout artifact(s) tagged and excluded: {tags}",
                )

            if gt.get("expect_tables"):
                has_tables = len(real_tables) > 0
                report.add(
                    "table-extraction",
                    short_name,
                    has_tables,
                    f"Expected tables — found {len(real_tables)}",
                    severity="MAJOR",
                )
                # Check table content quality (non-empty cells) — skip artifacts
                for i, tab in enumerate(extraction.tables):
                    if tab.artifact_type:
                        continue
                    non_empty = sum(1 for row in tab.rows for cell in row if cell.strip())
                    total_cells = sum(len(row) for row in tab.rows)
                    if total_cells > 0:
                        fill_rate = non_empty / total_cells
                        report.add(
                            "table-content-quality",
                            f"{short_name}/table-{i}",
                            fill_rate > 0.5,
                            f"Table {i}: {non_empty}/{total_cells} cells non-empty ({fill_rate:.0%}). "
                            f"Caption: '{(tab.caption or 'NONE')[:60]}'",
                            severity="MAJOR" if fill_rate < 0.2 else "MINOR",
                        )

            # --- 3c: Figures extracted ---
            if gt.get("expect_figures"):
                has_figures = len(extraction.figures) > 0
                report.add(
                    "figure-extraction",
                    short_name,
                    has_figures,
                    f"Expected figures — found {len(extraction.figures)}",
                    severity="MAJOR",
                )
                # Caption rate (informational — orphans are caught as MAJOR by 3d.3)
                captioned = [f for f in extraction.figures if f.caption]
                if extraction.figures:
                    caption_rate = len(captioned) / len(extraction.figures)
                    report.add(
                        "figure-caption-rate",
                        short_name,
                        caption_rate == 1.0,
                        f"{len(captioned)}/{len(extraction.figures)} figures have captions ({caption_rate:.0%})",
                        severity="MAJOR",
                    )

            # --- 3d: Completeness grade ---
            comp = extraction.completeness
            if comp:
                n_artifacts = len(artifact_tables)
                artifact_note = f" | Artifacts: {n_artifacts} tagged" if n_artifacts else ""
                report.add(
                    "completeness-grade",
                    short_name,
                    comp.grade in ("A", "B"),
                    f"Grade: {comp.grade} | "
                    f"Figs: {comp.figures_found} found / {comp.figure_captions_found} captioned / {comp.figures_missing} missing | "
                    f"Tables: {comp.tables_found} found / {comp.table_captions_found} captioned / {comp.tables_missing} missing"
                    + artifact_note,
                    severity="MAJOR" if comp.grade in ("D", "F") else "MINOR",
                )

                # --- 3d.1: Missing figures (captions found, no image extracted) ---
                if comp.figures_missing > 0:
                    report.add(
                        "missing-figures",
                        short_name,
                        False,
                        f"{comp.figures_missing} figure(s) have captions but no extracted image. "
                        f"Captions found: {comp.figure_captions_found}, figures extracted: {comp.figures_found}",
                        severity="MAJOR",
                    )

                # --- 3d.2: Missing tables (captions found, no table extracted) ---
                if comp.tables_missing > 0:
                    report.add(
                        "missing-tables",
                        short_name,
                        False,
                        f"{comp.tables_missing} table(s) have captions but no extracted content. "
                        f"Captions found: {comp.table_captions_found}, tables extracted: {comp.tables_found}",
                        severity="MAJOR",
                    )

                # --- 3d.3: Orphan figures (extracted but no caption matched) ---
                orphan_figs = comp.figures_found - comp.figures_with_captions
                if orphan_figs > 0:
                    report.add(
                        "orphan-figures",
                        short_name,
                        False,
                        f"{orphan_figs} figure(s) extracted without a real caption. "
                        f"Unmatched caption numbers: {comp.unmatched_figure_captions or 'none'}",
                        severity="MAJOR",
                    )

                # --- 3d.4: Orphan tables (extracted but no caption matched) ---
                orphan_tabs = comp.tables_found - comp.tables_with_captions
                if orphan_tabs > 0:
                    report.add(
                        "orphan-tables",
                        short_name,
                        False,
                        f"{orphan_tabs} table(s) extracted without a real caption. "
                        f"Unmatched caption numbers: {comp.unmatched_table_captions or 'none'}",
                        severity="MAJOR",
                    )

                # --- 3d.5: Unmatched captions (caption on page, not on any object) ---
                unmatched = comp.unmatched_figure_captions + comp.unmatched_table_captions
                if unmatched:
                    report.add(
                        "unmatched-captions",
                        short_name,
                        False,
                        f"Caption numbers found on pages but not matched to any extracted object: "
                        f"figures={comp.unmatched_figure_captions or 'none'}, "
                        f"tables={comp.unmatched_table_captions or 'none'}",
                        severity="MAJOR",
                    )

            # --- 3e: Abstract detected ---
            has_abstract = any(s.label == "abstract" for s in extraction.sections)
            report.add(
                "abstract-detection",
                short_name,
                has_abstract,
                f"Abstract {'detected' if has_abstract else 'NOT detected'}",
                severity="MINOR",
            )

            # --- 3h: Content readability (garbled/interleaved cells) ---
            if comp and extraction.tables:
                from deep_zotero.pdf_processor import _check_content_readability
                readability_issues = []
                for ti, tab in enumerate(extraction.tables):
                    rpt = _check_content_readability(tab)
                    if rpt["garbled_cells"] or rpt["interleaved_cells"]:
                        readability_issues.append(
                            f"table {ti}: garbled={rpt['garbled_cells']}, "
                            f"interleaved={rpt['interleaved_cells']}"
                        )
                report.add(
                    "content-readability",
                    short_name,
                    len(readability_issues) == 0,
                    f"{len(readability_issues)} tables with readability issues"
                    + (f": {'; '.join(readability_issues[:3])}" if readability_issues else ""),
                    severity="MAJOR",
                )

            # --- 3i: 1x1 table dimensions ---
            if comp:
                report.add(
                    "table-dimensions-sanity",
                    short_name,
                    comp.tables_1x1 == 0,
                    f"{comp.tables_1x1} tables are 1x1 (degenerate)",
                    severity="MAJOR",
                )

            # --- 3j: Caption encoding quality ---
            if comp:
                report.add(
                    "caption-encoding-quality",
                    short_name,
                    comp.encoding_artifact_captions == 0,
                    f"{comp.encoding_artifact_captions} captions with encoding artifacts",
                    severity="MINOR",
                )

            # --- 3k: Caption number continuity ---
            if comp:
                gaps = comp.figure_number_gaps + comp.table_number_gaps
                report.add(
                    "caption-number-continuity",
                    short_name,
                    len(gaps) == 0,
                    f"Figure gaps: {comp.figure_number_gaps or 'none'}, "
                    f"Table gaps: {comp.table_number_gaps or 'none'}",
                    severity="MAJOR",
                )

            # --- 3l: Duplicate captions ---
            if comp:
                report.add(
                    "duplicate-captions",
                    short_name,
                    comp.duplicate_captions == 0,
                    f"{comp.duplicate_captions} duplicate caption(s) found",
                    severity="MAJOR",
                )

            # --- 3f: Chunk count sanity ---
            expected_min_chunks = len(extraction.pages) * 2  # At least 2 chunks per page
            report.add(
                "chunk-count-sanity",
                short_name,
                len(chunks) >= expected_min_chunks,
                f"{len(chunks)} chunks for {len(extraction.pages)} pages "
                f"(expected >= {expected_min_chunks})",
                severity="MAJOR" if len(chunks) < len(extraction.pages) else "MINOR",
            )

            # --- 3g: Check for image files written ---
            if gt.get("expect_figures") and extraction.figures:
                figs_with_images = [f for f in extraction.figures if f.image_path and f.image_path.exists()]
                report.add(
                    "figure-images-saved",
                    short_name,
                    len(figs_with_images) > 0,
                    f"{len(figs_with_images)}/{len(extraction.figures)} figure images saved to disk",
                    severity="MINOR",
                )

        # ===================================================================
        # PHASE 4: Semantic search tests (researcher workflow)
        # ===================================================================
        print(f"\n[PHASE 4] Running semantic search tests...")

        for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
            query = gt["searchable_content"]
            print(f"  Searching for [{short_name}]: '{query[:50]}...'")

            t0 = time.perf_counter()
            results = retriever.search(query=query, top_k=10, context_window=1)
            t_search = time.perf_counter() - t0

            # Did the target paper appear in results?
            target_hits = [r for r in results if r.doc_id == item_key]
            found = len(target_hits) > 0
            rank = None
            if found:
                for i, r in enumerate(results):
                    if r.doc_id == item_key:
                        rank = i + 1
                        break

            report.add(
                "semantic-search-recall",
                short_name,
                found,
                f"Query: '{query[:50]}' — "
                + (f"found at rank {rank}/10 (score {target_hits[0].score:.3f})"
                   if found else f"NOT FOUND in top 10. Got: {[r.doc_title[:30] for r in results[:3]]}"),
                severity="MAJOR",
            )

            # Check if it ranks in top 3 (a researcher would expect this)
            if found:
                report.add(
                    "semantic-search-ranking",
                    short_name,
                    rank <= 3,
                    f"Ranked {rank}/10 for its own core content query",
                    severity="MAJOR" if rank > 5 else "MINOR",
                )

        # ===================================================================
        # PHASE 5: Table search tests
        # ===================================================================
        print(f"\n[PHASE 5] Running table search tests...")

        for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
            table_query = gt.get("table_search_query")
            if not table_query:
                continue
            if not extraction.tables:
                continue

            print(f"  Searching tables for [{short_name}]: '{table_query}'")

            # Search tables
            table_filter = {"chunk_type": {"$eq": "table"}}
            table_results = store.search(query=table_query, top_k=10, filters=table_filter)

            target_tables = [r for r in table_results if r.metadata.get("doc_id") == item_key]
            found = len(target_tables) > 0

            report.add(
                "table-search-recall",
                short_name,
                found,
                f"Query: '{table_query}' — "
                + (f"found {len(target_tables)} matching table(s), "
                   f"best score {target_tables[0].score:.3f}, "
                   f"caption: '{target_tables[0].metadata.get('table_caption', 'NONE')[:50]}'"
                   if found else f"NOT FOUND. Got: {[r.metadata.get('doc_title', '?')[:30] for r in table_results[:3]]}"),
                severity="MAJOR",
            )

            # Check table markdown content quality
            if found:
                best_table = target_tables[0]
                table_text = best_table.text
                has_pipe = "|" in table_text
                has_rows = table_text.count("\n") >= 2
                report.add(
                    "table-markdown-quality",
                    short_name,
                    has_pipe and has_rows,
                    f"Table markdown has {'pipes' if has_pipe else 'NO pipes'} and "
                    f"{table_text.count(chr(10))} lines. "
                    f"Preview: {table_text[:100].replace(chr(10), ' | ')}",
                    severity="MINOR",
                )

        # ===================================================================
        # PHASE 6: Figure search tests
        # ===================================================================
        print(f"\n[PHASE 6] Running figure search tests...")

        for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
            fig_query = gt.get("figure_search_query")
            if not fig_query:
                continue
            if not extraction.figures:
                continue

            print(f"  Searching figures for [{short_name}]: '{fig_query}'")

            fig_filter = {"chunk_type": {"$eq": "figure"}}
            fig_results = store.search(query=fig_query, top_k=10, filters=fig_filter)

            target_figs = [r for r in fig_results if r.metadata.get("doc_id") == item_key]
            found = len(target_figs) > 0

            report.add(
                "figure-search-recall",
                short_name,
                found,
                f"Query: '{fig_query}' — "
                + (f"found {len(target_figs)} matching figure(s), "
                   f"best score {target_figs[0].score:.3f}, "
                   f"caption: '{target_figs[0].metadata.get('caption', 'NONE')[:50]}'"
                   if found else f"NOT FOUND in top 10. "
                   f"Got: {[r.metadata.get('doc_title', '?')[:30] for r in fig_results[:3]]}"),
                severity="MAJOR",
            )

        # ===================================================================
        # PHASE 7: Metadata filter tests
        # ===================================================================
        print(f"\n[PHASE 7] Testing metadata filters...")

        # Test author filter using _apply_text_filters (Fix 1 added _meta_get helper)
        from deep_zotero.server import _apply_text_filters
        for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
            author_substr = gt.get("author_substr", "")
            if not author_substr:
                continue

            query = gt["searchable_content"]
            results = retriever.search(query=query, top_k=50, context_window=0)

            # Use the server's _apply_text_filters — Fix 1 made this work on RetrievalResult
            filtered = _apply_text_filters(results, author=author_substr)

            target_hits = [r for r in filtered if r.doc_id == item_key]
            found = len(target_hits) > 0

            report.add(
                "author-filter",
                short_name,
                found,
                f"Filter author='{author_substr}' — "
                + (f"target paper found ({len(filtered)} total results after filter)"
                   if found else f"target paper NOT found after filtering"),
                severity="MAJOR",
            )

        # Test year range filter
        print("  Testing year range filters...")
        year_filter = {"year": {"$gte": 2015}}
        results = retriever.search(
            query="heart rate variability",
            top_k=50,
            context_window=0,
            filters=year_filter,
        )
        old_papers = [r for r in results if r.year and r.year < 2015]
        report.add(
            "year-filter-accuracy",
            "all",
            len(old_papers) == 0,
            f"Year filter >=2015: {len(old_papers)} papers from before 2015 leaked through "
            f"(total results: {len(results)})",
            severity="MAJOR" if len(old_papers) > 0 else "MINOR",
        )

        # ===================================================================
        # PHASE 8: Context expansion tests
        # ===================================================================
        print(f"\n[PHASE 8] Testing context expansion...")

        for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
            query = gt["searchable_content"]
            results = retriever.search(query=query, top_k=5, context_window=2)

            target_hits = [r for r in results if r.doc_id == item_key]
            if not target_hits:
                continue

            hit = target_hits[0]
            has_context = bool(hit.context_before or hit.context_after)
            full_ctx = hit.full_context()

            report.add(
                "context-expansion",
                short_name,
                has_context,
                f"Context expansion: "
                + (f"before={len(hit.context_before)}, after={len(hit.context_after)}, "
                   f"full_context={len(full_ctx)} chars"
                   if has_context else "NO context returned"),
                severity="MINOR",
            )

            # Check context is longer than the hit alone
            if has_context:
                report.add(
                    "context-adds-value",
                    short_name,
                    len(full_ctx) > len(hit.text),
                    f"Full context ({len(full_ctx)} chars) vs hit ({len(hit.text)} chars)",
                    severity="MINOR",
                )

        # ===================================================================
        # PHASE 9: Cross-paper search (topic search simulation)
        # ===================================================================
        print(f"\n[PHASE 9] Testing cross-paper topic search...")

        # A researcher searches for "heart rate variability autonomic" — should find
        # multiple relevant papers
        topic_query = "heart rate variability autonomic nervous system"
        print(f"  Topic: '{topic_query}'")
        results = retriever.search(query=topic_query, top_k=50, context_window=0)

        # Rerank
        from dataclasses import replace
        reranked = reranker.rerank(results)

        # Group by document
        from collections import defaultdict
        by_doc = defaultdict(list)
        for r in reranked:
            by_doc[r.doc_id].append(r)

        hrv_keys = {"9GKLLJH9", "AQ3D94VC"}  # helm-coregulation, reyes-lf-hrv
        found_hrv = hrv_keys & set(by_doc.keys())
        report.add(
            "topic-search-multi-paper",
            "HRV papers",
            len(found_hrv) >= 1,
            f"Topic search for HRV: found {len(found_hrv)}/{len(hrv_keys)} expected papers "
            f"in {len(by_doc)} total docs. Keys found: {found_hrv}",
            severity="MAJOR",
        )

        # Another cross-domain search
        topic_query2 = "electrode impedance measurement skin contact"
        print(f"  Topic: '{topic_query2}'")
        results2 = retriever.search(query=topic_query2, top_k=50, context_window=0)
        by_doc2 = defaultdict(list)
        for r in results2:
            by_doc2[r.doc_id].append(r)

        impedance_keys = {"VP3NJ74M"}  # fortune-impedance
        found_imp = impedance_keys & set(by_doc2.keys())
        report.add(
            "topic-search-engineering",
            "impedance papers",
            len(found_imp) >= 1,
            f"Topic search for impedance: found {len(found_imp)}/{len(impedance_keys)} expected. "
            f"Total docs: {len(by_doc2)}. Keys: {found_imp}",
            severity="MAJOR",
        )

        # ===================================================================
        # PHASE 10: OCR pathway test
        # ===================================================================
        print(f"\n[PHASE 10] Testing OCR pathway...")

        # Convert first page of one paper to image, make a new PDF from it,
        # and try to extract text
        import pymupdf

        ocr_path.mkdir(parents=True, exist_ok=True)

        # Pick a paper with clean text
        test_item = None
        test_short = None
        for ik, (ext, ch, it, gt, sn) in extractions.items():
            if len(ext.pages) >= 5:
                test_item = it
                test_short = sn
                break

        if test_item:
            print(f"  Converting [{test_short}] pages 1-3 to images...")
            src_doc = pymupdf.open(str(test_item.pdf_path))

            # Render pages to images at moderate DPI (simulates scanned doc)
            img_pdf_path = ocr_path / "ocr_test.pdf"
            img_doc = pymupdf.open()  # New empty PDF

            for page_idx in range(min(3, len(src_doc))):
                page = src_doc[page_idx]
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")

                # Create a new page from the image
                img_page = img_doc.new_page(
                    width=page.rect.width,
                    height=page.rect.height,
                )
                img_page.insert_image(
                    img_page.rect,
                    stream=img_bytes,
                )

            img_doc.save(str(img_pdf_path))
            img_doc.close()
            src_doc.close()

            print(f"  Extracting text from image-only PDF...")
            ocr_extraction = extract_document(img_pdf_path)

            total_text = sum(len(p.markdown) for p in ocr_extraction.pages)
            ocr_pages = ocr_extraction.stats.get("ocr_pages", 0)

            report.add(
                "ocr-text-extraction",
                test_short,
                total_text > 100,
                f"OCR extracted {total_text} chars from {len(ocr_extraction.pages)} image pages. "
                f"OCR pages detected: {ocr_pages}",
                severity="MAJOR",
            )

            report.add(
                "ocr-page-detection",
                test_short,
                ocr_pages > 0,
                f"OCR page detection: {ocr_pages}/{len(ocr_extraction.pages)} pages flagged as OCR",
                severity="MINOR",
            )
        else:
            raise RuntimeError("No suitable paper found for OCR test")

        # ===================================================================
        # PHASE 11: Edge case tests
        # ===================================================================
        print(f"\n[PHASE 11] Testing edge cases...")

        # Test: search with zero results
        nonsense_results = retriever.search(
            query="xylophone quantum superconductor banana",
            top_k=5,
            context_window=0,
        )
        # These should return low-score results, not crash
        top_score = f"{nonsense_results[0].score:.3f}" if nonsense_results else "N/A"
        report.add(
            "nonsense-query-no-crash",
            "all",
            nonsense_results[0].score < 0.5 if nonsense_results else False,
            f"Nonsense query returned {len(nonsense_results)} results "
            f"(top score: {top_score})",
            severity="MINOR",
        )

        # Test: reranker doesn't crash on empty results
        try:
            empty_reranked = reranker.rerank([])
            report.add(
                "empty-rerank-no-crash",
                "all",
                True,
                "Reranker handles empty input gracefully",
                severity="MINOR",
            )
        except Exception as e:
            report.add(
                "empty-rerank-no-crash",
                "all",
                False,
                f"Reranker crashed on empty input: {e}",
                severity="MINOR",
            )

        # Test: adjacent chunk retrieval for first/last chunks
        for item_key in list(extractions.keys())[:2]:
            ext, chunks_list, item, gt, sn = extractions[item_key]
            if chunks_list:
                # First chunk — should not crash
                adj = store.get_adjacent_chunks(item_key, 0, window=2)
                report.add(
                    "boundary-chunk-first",
                    sn,
                    len(adj) >= 1,
                    f"Adjacent chunks for first chunk: got {len(adj)} (expected >=1)",
                    severity="MINOR",
                )
                # Last chunk
                last_idx = chunks_list[-1].chunk_index
                adj_last = store.get_adjacent_chunks(item_key, last_idx, window=2)
                report.add(
                    "boundary-chunk-last",
                    sn,
                    len(adj_last) >= 1,
                    f"Adjacent chunks for last chunk (idx={last_idx}): got {len(adj_last)}",
                    severity="MINOR",
                )

        # ===================================================================
        # PHASE 12: Section-weighted search tests
        # ===================================================================
        print(f"\n[PHASE 12] Testing section-weighted reranking...")

        # Search with methods section boosted
        methods_query = "experimental protocol measurement procedure"
        results_default = retriever.search(query=methods_query, top_k=20, context_window=0)
        reranked_default = reranker.rerank(results_default)

        methods_weights = {"methods": 1.5, "results": 1.0, "abstract": 0.5}
        reranked_methods = reranker.rerank(results_default, section_weights=methods_weights)

        if reranked_default and reranked_methods:
            # Methods-boosted reranking should change the order
            default_top3_sections = [r.section for r in reranked_default[:3]]
            methods_top3_sections = [r.section for r in reranked_methods[:3]]
            order_changed = default_top3_sections != methods_top3_sections
            report.add(
                "section-weight-effect",
                "all",
                order_changed,
                f"Default top-3 sections: {default_top3_sections}, "
                f"methods-boosted top-3: {methods_top3_sections}, "
                f"order changed: {order_changed}",
                severity="MINOR",
            )

        # ===================================================================
        # PHASE 13: Validate section labels are consistent
        # ===================================================================
        print(f"\n[PHASE 13] Validating section label consistency...")

        valid_labels = {
            "abstract", "introduction", "background", "methods", "results",
            "discussion", "conclusion", "references", "appendix",
            "preamble", "unknown",
        }

        for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
            for section in extraction.sections:
                if section.label not in valid_labels:
                    report.add(
                        "invalid-section-label",
                        short_name,
                        False,
                        f"Invalid section label '{section.label}' in heading '{section.heading_text}'",
                        severity="MAJOR",
                    )
                    break
            else:
                report.add(
                    "section-labels-valid",
                    short_name,
                    True,
                    f"All {len(extraction.sections)} section labels are valid",
                    severity="MINOR",
                )

            # Check sections cover the full document
            if extraction.sections:
                first_start = extraction.sections[0].char_start
                last_end = extraction.sections[-1].char_end
                total_len = len(extraction.full_markdown)
                coverage = (last_end - first_start) / total_len if total_len > 0 else 0

                report.add(
                    "section-coverage",
                    short_name,
                    coverage >= 0.9,
                    f"Section spans cover {coverage:.0%} of document "
                    f"(first: {first_start}, last: {last_end}, total: {total_len})",
                    severity="MINOR",
                )

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    return report, extractions


# ---------------------------------------------------------------------------
# Debug database — structured SQLite for agent-friendly artifact inspection
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    item_key              TEXT PRIMARY KEY,
    short_name            TEXT NOT NULL,
    title                 TEXT,
    num_pages             INTEGER,
    num_chunks            INTEGER,
    quality_grade         TEXT,
    -- completeness fields (NULL when completeness unavailable)
    figures_found         INTEGER,
    figures_with_captions INTEGER,
    figures_missing       INTEGER,
    figure_captions_found INTEGER,
    tables_found          INTEGER,
    tables_with_captions  INTEGER,
    tables_missing        INTEGER,
    table_captions_found  INTEGER,
    tables_1x1            INTEGER,
    encoding_artifact_captions INTEGER,
    duplicate_captions    INTEGER,
    figure_number_gaps    TEXT,   -- JSON array of gap strings
    table_number_gaps     TEXT,   -- JSON array of gap strings
    unmatched_figure_captions TEXT, -- JSON array
    unmatched_table_captions  TEXT, -- JSON array
    completeness_grade    TEXT,
    full_markdown         TEXT,   -- full extracted document text
    pdf_path              TEXT    -- path to source PDF (for debug viewer)
);

CREATE TABLE IF NOT EXISTS sections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key       TEXT NOT NULL,
    section_index  INTEGER,
    label          TEXT,
    heading_text   TEXT,
    char_start     INTEGER,
    char_end       INTEGER,
    confidence     REAL,
    FOREIGN KEY (item_key) REFERENCES papers(item_key)
);

CREATE TABLE IF NOT EXISTS pages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key  TEXT NOT NULL,
    page_num  INTEGER,
    markdown  TEXT,
    FOREIGN KEY (item_key) REFERENCES papers(item_key)
);

CREATE TABLE IF NOT EXISTS extracted_tables (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key          TEXT NOT NULL,
    table_index       INTEGER,
    page_num          INTEGER,
    caption           TEXT,
    caption_position  TEXT,
    num_rows          INTEGER,
    num_cols          INTEGER,
    non_empty_cells   INTEGER,
    total_cells       INTEGER,
    fill_rate         REAL,
    headers_json      TEXT,   -- JSON array of header strings
    rows_json         TEXT,   -- JSON array of arrays (full cell data)
    markdown          TEXT,   -- rendered markdown table
    reference_context TEXT,
    bbox              TEXT,   -- JSON [x0, y0, x1, y1]
    artifact_type     TEXT,   -- NULL=real data, else layout artifact tag
    extraction_strategy TEXT, -- which multi-strategy winner produced cell text
    table_id            TEXT,   -- stable table ID for linking
    FOREIGN KEY (item_key) REFERENCES papers(item_key)
);

CREATE TABLE IF NOT EXISTS extracted_figures (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key          TEXT NOT NULL,
    figure_index      INTEGER,
    page_num          INTEGER,
    caption           TEXT,
    bbox              TEXT,   -- JSON [x0, y0, x1, y1]
    image_path        TEXT,
    has_image         INTEGER,
    reference_context TEXT,
    FOREIGN KEY (item_key) REFERENCES papers(item_key)
);

CREATE TABLE IF NOT EXISTS chunks (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key             TEXT NOT NULL,
    chunk_index          INTEGER,
    page_num             INTEGER,
    section              TEXT,
    section_confidence   REAL,
    char_start           INTEGER,
    char_end             INTEGER,
    text                 TEXT,
    FOREIGN KEY (item_key) REFERENCES papers(item_key)
);

CREATE TABLE IF NOT EXISTS test_results (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT,
    paper     TEXT,
    passed    INTEGER,
    detail    TEXT,
    severity  TEXT
);
"""


def _make_simple_table_id(item_key: str, caption: str | None, page_num: int, index: int) -> str:
    """Generate a simple table ID from caption or page/index.

    This is a local replacement for the removed ground_truth.make_table_id.
    It produces stable IDs for the debug DB without requiring the GT module.
    """
    if caption:
        import re as _re
        m = _re.search(r"(?:Table|Tab\.?)\s+(\d+)", caption, _re.IGNORECASE)
        if m:
            return f"{item_key}_table_{m.group(1)}"
    return f"{item_key}_orphan_p{page_num}_t{index}"


def write_debug_database(
    extractions: dict[str, tuple],
    report: StressTestReport,
    db_path: Path,
) -> None:
    """Write all extraction artifacts and test results to a SQLite database.

    Replaces the old text-file audit.  Agents can then query specific papers,
    tables, figures, chunks, or test failures via SQL instead of reading a
    giant text dump.
    """
    # Remove stale DB so each run is a clean snapshot
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    con.executescript(_SCHEMA)
    create_extended_tables(con)

    # --- run metadata ---
    meta_rows = [
        ("generated", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("corpus_size", str(len(CORPUS))),
        ("papers_extracted", str(len(extractions))),
        ("tests_total", str(len(report.results))),
        ("tests_passed", str(report.passed)),
        ("tests_failed", str(report.failed)),
        ("major_failures", str(report.major_failures)),
    ]
    for k, v in report.timings.items():
        meta_rows.append((f"timing_{k}", f"{v:.1f}s"))
    con.executemany(
        "INSERT INTO run_metadata (key, value) VALUES (?, ?)", meta_rows,
    )

    for item_key, (extraction, chunks, item, gt, short_name) in extractions.items():
        comp = extraction.completeness

        # --- paper row ---
        con.execute(
            """INSERT INTO papers VALUES (
                ?,?,?,?,?,?,  ?,?,?,?,  ?,?,?,?,  ?,?,?,  ?,?,?,?,  ?,?,?
            )""",
            (
                item_key,
                short_name,
                item.title,
                len(extraction.pages),
                len(chunks),
                extraction.quality_grade,
                # completeness (may be None)
                comp.figures_found if comp else None,
                comp.figures_with_captions if comp else None,
                comp.figures_missing if comp else None,
                comp.figure_captions_found if comp else None,
                comp.tables_found if comp else None,
                comp.tables_with_captions if comp else None,
                comp.tables_missing if comp else None,
                comp.table_captions_found if comp else None,
                comp.tables_1x1 if comp else None,
                comp.encoding_artifact_captions if comp else None,
                comp.duplicate_captions if comp else None,
                json.dumps(comp.figure_number_gaps) if comp else None,
                json.dumps(comp.table_number_gaps) if comp else None,
                json.dumps(comp.unmatched_figure_captions) if comp else None,
                json.dumps(comp.unmatched_table_captions) if comp else None,
                comp.grade if comp else None,
                extraction.full_markdown,
                str(item.pdf_path) if item.pdf_path else None,
            ),
        )

        # --- sections ---
        for si, sec in enumerate(extraction.sections):
            con.execute(
                "INSERT INTO sections (item_key, section_index, label, heading_text, "
                "char_start, char_end, confidence) VALUES (?,?,?,?,?,?,?)",
                (item_key, si, sec.label, sec.heading_text,
                 sec.char_start, sec.char_end, sec.confidence),
            )

        # --- pages ---
        for pg in extraction.pages:
            con.execute(
                "INSERT INTO pages (item_key, page_num, markdown) VALUES (?,?,?)",
                (item_key, pg.page_num, pg.markdown),
            )

        # --- tables ---
        for tab in extraction.tables:
            table_id = _make_simple_table_id(
                item_key, tab.caption, tab.page_num, tab.table_index
            )
            non_empty = sum(1 for row in tab.rows for cell in row if cell.strip())
            total_cells = sum(len(row) for row in tab.rows)
            fill_rate = non_empty / total_cells if total_cells else 0.0
            con.execute(
                "INSERT INTO extracted_tables (item_key, table_index, page_num, "
                "caption, caption_position, num_rows, num_cols, non_empty_cells, "
                "total_cells, fill_rate, headers_json, rows_json, markdown, "
                "reference_context, bbox, artifact_type, extraction_strategy, "
                "table_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item_key,
                    tab.table_index,
                    tab.page_num,
                    tab.caption,
                    tab.caption_position,
                    tab.num_rows,
                    tab.num_cols,
                    non_empty,
                    total_cells,
                    fill_rate,
                    json.dumps(tab.headers),
                    json.dumps(tab.rows),
                    tab.to_markdown(),
                    tab.reference_context,
                    json.dumps(list(tab.bbox)),
                    tab.artifact_type,
                    tab.extraction_strategy,
                    table_id,
                ),
            )

        # --- vision extraction details ---
        if extraction.vision_details:
            for vi, vd in enumerate(extraction.vision_details):
                table_id = _make_simple_table_id(
                    item_key, vd["text_layer_caption"], vd["page_num"], vi
                )
                write_vision_agent_result(
                    con,
                    table_id=table_id,
                    agent_idx=0,
                    model="claude-haiku-4-5-20251001",
                    raw_response=vd["raw_response"],
                    headers_json=json.dumps(vd["headers"]),
                    rows_json=json.dumps(vd["rows"]),
                    table_label=vd["table_label"],
                    is_incomplete=vd["is_incomplete"],
                    incomplete_reason=vd["incomplete_reason"],
                    parse_success=vd["parse_success"],
                    execution_time_ms=None,
                    agent_role="transcriber",
                    footnotes=vd["footnotes"],
                    cell_accuracy_pct=None,
                )
                write_vision_run_detail(con, table_id=table_id, details_dict=vd)

        # --- figures ---
        for fig in extraction.figures:
            has_img = 1 if (fig.image_path and fig.image_path.exists()) else 0
            con.execute(
                "INSERT INTO extracted_figures (item_key, figure_index, page_num, "
                "caption, bbox, image_path, has_image, reference_context) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    item_key,
                    fig.figure_index,
                    fig.page_num,
                    fig.caption,
                    json.dumps(list(fig.bbox)),
                    str(fig.image_path) if fig.image_path else None,
                    has_img,
                    fig.reference_context,
                ),
            )

        # --- chunks ---
        for ch in chunks:
            con.execute(
                "INSERT INTO chunks (item_key, chunk_index, page_num, section, "
                "section_confidence, char_start, char_end, text) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    item_key,
                    ch.chunk_index,
                    ch.page_num,
                    ch.section,
                    ch.section_confidence,
                    ch.char_start,
                    ch.char_end,
                    ch.text,
                ),
            )

    # --- test results ---
    for r in report.results:
        con.execute(
            "INSERT INTO test_results (test_name, paper, passed, detail, severity) "
            "VALUES (?,?,?,?,?)",
            (r.test_name, r.paper, 1 if r.passed else 0, r.detail, r.severity),
        )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Vision extraction report builder
# ---------------------------------------------------------------------------


def _build_vision_extraction_report(db_path: Path) -> list[str]:
    """Query vision_run_details and vision_agent_results from the debug DB.

    Returns markdown lines for the Vision Extraction Report section, or an
    empty list when no vision data was recorded (vision_api was None).
    """
    con = sqlite3.connect(str(db_path))
    try:
        count_row = con.execute(
            "SELECT COUNT(*) FROM vision_run_details"
        ).fetchone()
        if count_row[0] == 0:
            return []

        lines: list[str] = []
        lines.append("## Vision Extraction Report")
        lines.append("")

        # --- Summary metrics ---
        total = con.execute(
            "SELECT COUNT(*) FROM vision_run_details"
        ).fetchone()[0]
        parse_success = con.execute(
            "SELECT COUNT(*) FROM vision_run_details WHERE parse_success = 1"
        ).fetchone()[0]
        recropped = con.execute(
            "SELECT COUNT(*) FROM vision_run_details WHERE recropped = 1"
        ).fetchone()[0]
        incomplete = con.execute(
            "SELECT COUNT(*) FROM vision_run_details WHERE is_incomplete = 1"
        ).fetchone()[0]

        parse_pct = 100 * parse_success / total if total else 0.0
        recrop_pct = 100 * recropped / total if total else 0.0
        incomplete_pct = 100 * incomplete / total if total else 0.0

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Tables attempted | {total} |")
        lines.append(f"| Parse success | {parse_success} ({parse_pct:.1f}%) |")
        lines.append(f"| Re-crops performed | {recropped} ({recrop_pct:.1f}%) |")
        lines.append(f"| Incomplete tables | {incomplete} ({incomplete_pct:.1f}%) |")
        lines.append("")

        # --- Per-paper breakdown ---
        paper_rows = con.execute(
            """
            SELECT
                p.short_name,
                COUNT(vrd.table_id) AS total,
                SUM(vrd.parse_success) AS parsed,
                SUM(vrd.recropped) AS recrops,
                SUM(vrd.is_incomplete) AS incomplete
            FROM vision_run_details vrd
            LEFT JOIN extracted_tables et ON et.table_id = vrd.table_id
            LEFT JOIN papers p ON p.item_key = et.item_key
            GROUP BY p.short_name
            ORDER BY p.short_name
            """
        ).fetchall()

        if paper_rows:
            lines.append("### Per-paper breakdown")
            lines.append("")
            lines.append("| Paper | Tables | Parsed | Re-cropped | Incomplete |")
            lines.append("|-------|--------|--------|------------|------------|")
            for short_name, total_p, parsed_p, recrops_p, incomplete_p in paper_rows:
                label = short_name if short_name else "(unknown)"
                lines.append(
                    f"| {label} | {total_p} | {parsed_p} "
                    f"| {recrops_p} | {incomplete_p} |"
                )
            lines.append("")

        # --- Caption changes ---
        caption_rows = con.execute(
            """
            SELECT table_id, text_layer_caption, vision_caption
            FROM vision_run_details
            WHERE vision_caption IS NOT NULL
              AND vision_caption != ''
              AND vision_caption != text_layer_caption
            ORDER BY table_id
            """
        ).fetchall()

        if caption_rows:
            lines.append("### Caption changes (text-layer → vision)")
            lines.append("")
            lines.append("| Table ID | Text Layer | Vision |")
            lines.append("|----------|-----------|--------|")
            for table_id, text_cap, vision_cap in caption_rows:
                text_short = (text_cap or "")[:60].replace("|", "/")
                vision_short = (vision_cap or "")[:60].replace("|", "/")
                lines.append(f"| {table_id} | {text_short} | {vision_short} |")
            lines.append("")

        return lines
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Selective re-run helpers
# ---------------------------------------------------------------------------


def _load_corpus_items():
    """Load Zotero items for the corpus (phase 1 only)."""
    from deep_zotero.zotero_client import ZoteroClient
    from deep_zotero.config import Config

    config = Config.load()
    zotero = ZoteroClient(config.zotero_data_dir)
    all_items = zotero.get_all_items_with_pdfs()
    items_by_key = {i.item_key: i for i in all_items}

    corpus_items = []
    skipped = []
    for item_key, short_name, reason, gt in CORPUS:
        item = items_by_key.get(item_key)
        if item is None:
            skipped.append(f"  SKIP {item_key} ({short_name}): not in library")
            continue
        if not item.pdf_path or not item.pdf_path.exists():
            skipped.append(f"  SKIP {item_key} ({short_name}): PDF missing")
            continue
        corpus_items.append((item, short_name, gt))
    return corpus_items, skipped


def run_vision_only(paper_filter: str | None, db_path: Path):
    """Re-run vision API extraction, replacing old rows."""
    corpus_items, skipped = _load_corpus_items()
    for msg in skipped:
        print(msg)

    if paper_filter:
        corpus_items = [
            (item, sn, gt) for item, sn, gt in corpus_items
            if item.item_key == paper_filter
        ]
        if not corpus_items:
            print(f"ERROR: paper {paper_filter} not found in corpus")
            sys.exit(1)

    from deep_zotero.pdf_processor import extract_document, resolve_pending_vision

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    from deep_zotero.feature_extraction.vision_api import VisionAPI
    vision_api = VisionAPI(api_key=api_key)
    print(f"\nVision-only re-run: {len(corpus_items)} papers, model={vision_api._model}")

    # Clear old vision rows
    for item, sn, gt in corpus_items:
        deleted = clear_vision_results(str(db_path), item_key=item.item_key)
        if deleted:
            print(f"  Cleared {deleted} old vision rows for {sn}")

    # Extract documents (needed for vision specs)
    doc_data: dict[str, tuple] = {}
    for item, short_name, gt in corpus_items:
        print(f"  Extracting [{short_name}]...", end=" ", flush=True)
        extraction = extract_document(
            item.pdf_path,
            write_images=True,
            images_dir=Path(__file__).parent.parent / "_stress_test_figures" / item.item_key,
            vision_api=vision_api,
        )
        n_pending = len(extraction.pending_vision.specs) if extraction.pending_vision else 0
        print(f"{n_pending} vision specs")
        doc_data[item.item_key] = (extraction, item, short_name, gt)

    # Resolve vision batch
    total_specs = sum(
        len(ext.pending_vision.specs)
        for ext, _, _, _ in doc_data.values()
        if ext.pending_vision
    )
    print(f"  Resolving vision batch: {total_specs} specs...")
    resolve_pending_vision(
        {k: v[0] for k, v in doc_data.items()},
        vision_api,
    )

    # Write to DB
    with sqlite3.connect(str(db_path)) as con:
        for item_key, (extraction, item, short_name, gt) in doc_data.items():
            if not extraction.vision_details:
                continue

            for vi, vd in enumerate(extraction.vision_details):
                table_id = _make_simple_table_id(
                    item_key, vd["text_layer_caption"], vd["page_num"], vi
                )
                write_vision_agent_result(
                    con,
                    table_id=table_id,
                    agent_idx=0,
                    model="claude-haiku-4-5-20251001",
                    raw_response=vd["raw_response"],
                    headers_json=json.dumps(vd["headers"]),
                    rows_json=json.dumps(vd["rows"]),
                    table_label=vd["table_label"],
                    is_incomplete=vd["is_incomplete"],
                    incomplete_reason=vd["incomplete_reason"],
                    parse_success=vd["parse_success"],
                    execution_time_ms=None,
                    agent_role="transcriber",
                    footnotes=vd["footnotes"],
                    cell_accuracy_pct=None,
                )
                write_vision_run_detail(con, table_id=table_id, details_dict=vd)

            print(f"  {short_name}: {len(extraction.vision_details)} vision results written")

    print(f"\nDone. Vision results updated in {db_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="Stress test: real-world researcher workflow simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Selective re-run examples:
  # Re-run vision API for all papers
  python stress_test_real_library.py --vision-only

  # Re-run vision API for a single paper
  python stress_test_real_library.py --vision-only --paper SCPXVBLY
""",
    )
    p.add_argument(
        "--vision-only", action="store_true",
        help="Only re-run vision API extraction (requires ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--paper",
        help="Only process this paper (item_key, e.g. SCPXVBLY)",
    )
    p.add_argument(
        "--db", type=Path,
        default=Path(__file__).parent.parent / "_stress_test_debug.db",
        help="Path to debug database (default: _stress_test_debug.db)",
    )
    return p


if __name__ == "__main__":
    # Fix Windows console encoding
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    args = _build_argparser().parse_args()

    # --- Mode: vision-only re-run ---
    if args.vision_only:
        if not args.db.exists():
            print(f"ERROR: database not found at {args.db}")
            sys.exit(1)
        run_vision_only(args.paper, args.db)
        sys.exit(0)

    # --- Mode: full stress test (default) ---
    print("=" * 70)
    print("  STRESS TEST: deep-zotero")
    print("  Real papers, real searches, real expectations")
    print("=" * 70)

    report, extractions = run_stress_test()

    base_dir = Path(__file__).parent.parent
    report_path = base_dir / "STRESS_TEST_REPORT.md"

    # Write debug database first so db_path can be set on report before rendering
    if extractions:
        db_path = args.db
        write_debug_database(extractions, report, db_path)
        print(f"Debug database saved to: {db_path} ({db_path.stat().st_size:,} bytes)")
        print(f"  Query with: sqlite3 {db_path} \"SELECT short_name, quality_grade FROM papers\"")
        print(f"  Tables: run_metadata, papers, sections, pages, extracted_tables, extracted_figures, chunks, test_results")
        report.db_path = db_path

    # Render report (includes vision section when db_path is set)
    md = report.to_markdown()
    print("\n" + "=" * 70)
    print(md)

    # Save report
    report_path.write_text(md, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Also print vision section to stdout separately for visibility
    if extractions:
        vision_md_lines = _build_vision_extraction_report(report.db_path)
        if vision_md_lines:
            print("\n".join(vision_md_lines))


    # Exit with appropriate code
    if report.major_failures > 0:
        print(f"\n*** {report.major_failures} MAJOR FAILURES — tool is unreliable ***")
        sys.exit(1)
    elif report.failed > 0:
        print(f"\n* {report.failed} minor issues found *")
        sys.exit(0)
    else:
        print("\nAll tests passed.")
        sys.exit(0)
