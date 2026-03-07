"""Debug database interface for recording and inspecting table extraction results."""
from __future__ import annotations

import dataclasses
import json
import sqlite3

from zotero_chunk_rag.feature_extraction.ground_truth import ComparisonResult


PADDLE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS paddle_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id    TEXT,
    page_num    INTEGER,
    engine_name TEXT,
    caption     TEXT,
    is_orphan   INTEGER,
    headers_json TEXT,
    rows_json   TEXT,
    bbox        TEXT,
    page_size   TEXT,
    raw_output  TEXT,
    item_key    TEXT
);

CREATE TABLE IF NOT EXISTS paddle_gt_diffs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id          TEXT,
    engine_name       TEXT,
    cell_accuracy_pct REAL,
    fuzzy_accuracy_pct REAL,
    num_splits        INTEGER,
    num_merges        INTEGER,
    num_cell_diffs    INTEGER,
    gt_shape          TEXT,
    ext_shape         TEXT,
    diff_json         TEXT
);
"""

EXTENDED_SCHEMA = """\
CREATE TABLE IF NOT EXISTS ground_truth_diffs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id            TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    diff_json           TEXT NOT NULL,
    cell_accuracy_pct   REAL,
    num_splits          INTEGER,
    num_merges          INTEGER,
    num_cell_diffs      INTEGER,
    gt_shape            TEXT,
    ext_shape           TEXT,
    fuzzy_accuracy_pct  REAL,
    fuzzy_precision_pct REAL,
    fuzzy_recall_pct    REAL
);

CREATE TABLE IF NOT EXISTS vision_agent_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id          TEXT NOT NULL,
    agent_idx         INTEGER NOT NULL,
    agent_role        TEXT,
    model             TEXT NOT NULL,
    raw_response      TEXT,
    headers_json      TEXT,
    rows_json         TEXT,
    table_label       TEXT,
    is_incomplete     INTEGER,
    incomplete_reason TEXT,
    parse_success     INTEGER,
    execution_time_ms INTEGER,
    corrections_json  TEXT,
    num_corrections   INTEGER,
    cell_accuracy_pct REAL,
    footnotes         TEXT
);

CREATE TABLE IF NOT EXISTS vision_run_details (
    table_id            TEXT PRIMARY KEY,
    text_layer_caption  TEXT,
    vision_caption      TEXT,
    page_num            INTEGER,
    crop_bbox_json      TEXT,
    recropped           BOOLEAN DEFAULT 0,
    recrop_bbox_pct_json TEXT,
    parse_success       BOOLEAN,
    is_incomplete       BOOLEAN,
    incomplete_reason   TEXT,
    recrop_needed       BOOLEAN,
    raw_response        TEXT,
    headers_json        TEXT,
    rows_json           TEXT,
    footnotes           TEXT,
    table_label         TEXT,
    fullpage_attempted  BOOLEAN DEFAULT 0,
    fullpage_parse_success BOOLEAN
);
"""


def create_extended_tables(con: sqlite3.Connection) -> None:
    """Execute the extended schema on an existing connection.

    Safe to call multiple times — all statements use CREATE TABLE IF NOT EXISTS.
    """
    con.executescript(EXTENDED_SCHEMA)
    con.executescript(PADDLE_SCHEMA)


def write_ground_truth_diff(
    con: sqlite3.Connection,
    table_id: str,
    run_id: str,
    comparison_result: ComparisonResult,
) -> None:
    """Serialize a ComparisonResult and insert a ground truth diff row.

    ``diff_json`` contains the full ComparisonResult serialized via
    ``dataclasses.asdict()`` + ``json.dumps()``.  Summary fields are extracted
    directly from the ComparisonResult for easy querying without JSON parsing.
    """
    diff_dict = dataclasses.asdict(comparison_result)
    diff_json = json.dumps(diff_dict, ensure_ascii=False)

    num_splits = (
        len(comparison_result.row_splits) + len(comparison_result.column_splits)
    )
    num_merges = (
        len(comparison_result.row_merges) + len(comparison_result.column_merges)
    )
    num_cell_diffs = len(comparison_result.cell_diffs)
    gt_shape = json.dumps(list(comparison_result.gt_shape))
    ext_shape = json.dumps(list(comparison_result.ext_shape))

    con.execute(
        "INSERT INTO ground_truth_diffs "
        "(table_id, run_id, diff_json, cell_accuracy_pct, num_splits, "
        "num_merges, num_cell_diffs, gt_shape, ext_shape, "
        "fuzzy_accuracy_pct, fuzzy_precision_pct, fuzzy_recall_pct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            table_id,
            run_id,
            diff_json,
            comparison_result.cell_accuracy_pct,
            num_splits,
            num_merges,
            num_cell_diffs,
            gt_shape,
            ext_shape,
            comparison_result.fuzzy_accuracy_pct,
            comparison_result.fuzzy_precision_pct,
            comparison_result.fuzzy_recall_pct,
        ),
    )


def write_vision_agent_result(
    con: sqlite3.Connection,
    table_id: str,
    agent_idx: int,
    model: str,
    raw_response: str | None,
    headers_json: str | None,
    rows_json: str | None,
    table_label: str | None,
    is_incomplete: bool,
    incomplete_reason: str,
    parse_success: bool,
    execution_time_ms: int | None,
    agent_role: str | None = None,
    corrections_json: str | None = None,
    num_corrections: int | None = None,
    cell_accuracy_pct: float | None = None,
    footnotes: str | None = None,
) -> None:
    """Insert a single vision agent result row."""
    con.execute(
        "INSERT INTO vision_agent_results "
        "(table_id, agent_idx, agent_role, model, raw_response, headers_json, rows_json, "
        "table_label, is_incomplete, incomplete_reason, parse_success, execution_time_ms, "
        "corrections_json, num_corrections, cell_accuracy_pct, footnotes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (table_id, agent_idx, agent_role, model, raw_response, headers_json, rows_json,
         table_label, int(is_incomplete), incomplete_reason, int(parse_success),
         execution_time_ms, corrections_json, num_corrections, cell_accuracy_pct,
         footnotes),
    )


def write_vision_run_detail(
    con: sqlite3.Connection,
    *,
    table_id: str,
    details_dict: dict,
) -> None:
    """Insert or replace a vision run detail row from a details dict.

    Uses INSERT OR REPLACE for idempotency — writing the same table_id twice
    overwrites the previous entry.

    The details_dict must contain the keys defined by the vision_details schema
    in Task 4.3.1.
    """
    crop_bbox = details_dict.get("crop_bbox")
    recrop_bbox_pct = details_dict.get("recrop_bbox_pct")
    fullpage_ps = details_dict.get("fullpage_parse_success")
    con.execute(
        "INSERT OR REPLACE INTO vision_run_details "
        "(table_id, text_layer_caption, vision_caption, page_num, crop_bbox_json, "
        "recropped, recrop_bbox_pct_json, parse_success, is_incomplete, "
        "incomplete_reason, recrop_needed, raw_response, headers_json, rows_json, "
        "footnotes, table_label, fullpage_attempted, fullpage_parse_success) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            table_id,
            details_dict.get("text_layer_caption"),
            details_dict.get("vision_caption"),
            details_dict.get("page_num"),
            json.dumps(crop_bbox) if crop_bbox is not None else None,
            int(bool(details_dict.get("recropped", False))),
            json.dumps(recrop_bbox_pct) if recrop_bbox_pct is not None else None,
            int(bool(details_dict.get("parse_success", False))),
            int(bool(details_dict.get("is_incomplete", False))),
            details_dict.get("incomplete_reason"),
            int(bool(details_dict.get("recrop_needed", False))),
            details_dict.get("raw_response"),
            json.dumps(details_dict.get("headers", [])),
            json.dumps(details_dict.get("rows", [])),
            details_dict.get("footnotes"),
            details_dict.get("table_label"),
            int(bool(details_dict.get("fullpage_attempted", False))),
            int(fullpage_ps) if fullpage_ps is not None else None,
        ),
    )


def clear_paddle_results(
    db_path: str,
    engine_name: str | None = None,
    item_key: str | None = None,
) -> int:
    """Delete paddle rows matching the given scope.

    Returns the total number of deleted rows across both tables.
    """
    with sqlite3.connect(db_path) as con:
        con.executescript(PADDLE_SCHEMA)
        deleted = 0
        for table in ("paddle_results", "paddle_gt_diffs"):
            clauses: list[str] = []
            params: list[str] = []
            if engine_name:
                clauses.append("engine_name = ?")
                params.append(engine_name)
            if item_key:
                col = "item_key" if table == "paddle_results" else "table_id"
                if table == "paddle_gt_diffs":
                    clauses.append("table_id LIKE ?")
                    params.append(f"{item_key}%")
                else:
                    clauses.append("item_key = ?")
                    params.append(item_key)
            where = " AND ".join(clauses) if clauses else "1=1"
            cur = con.execute(f"DELETE FROM {table} WHERE {where}", params)
            deleted += cur.rowcount
    return deleted


def clear_vision_results(
    db_path: str,
    item_key: str | None = None,
) -> int:
    """Delete vision rows matching the given scope.

    Returns the total number of deleted rows across both tables.
    """
    with sqlite3.connect(db_path) as con:
        deleted = 0
        for table in ("vision_agent_results", "vision_run_details"):
            try:
                if item_key:
                    cur = con.execute(
                        f"DELETE FROM {table} WHERE table_id LIKE ?",
                        (f"{item_key}%",),
                    )
                else:
                    cur = con.execute(f"DELETE FROM {table}")
                deleted += cur.rowcount
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet
    return deleted


def fix_duplicate_table_ids(db_path: str) -> dict[str, int]:
    """Disambiguate colliding table_ids in extracted_tables, paddle, and vision tables.

    For each table, groups rows by the natural dedup key (item_key or
    table_id prefix), detects collisions, and appends ``_p{page}`` to
    disambiguate.  Returns ``{table_name: rows_updated}``.
    """
    import re

    stats: dict[str, int] = {}

    with sqlite3.connect(db_path) as con:
        # --- extracted_tables: group by item_key ---
        try:
            rows = con.execute(
                "SELECT rowid, table_id, page_num, item_key "
                "FROM extracted_tables ORDER BY item_key, page_num, rowid"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        updates = 0
        et_groups: dict[str, list[tuple[int, str, int]]] = {}
        for row_id, tid, pn, ik in rows:
            et_groups.setdefault(ik, []).append((row_id, tid, pn))

        for _ik, group in et_groups.items():
            from collections import Counter
            id_counts = Counter(tid for _, tid, _ in group)
            for row_id, tid, pn in group:
                if id_counts[tid] > 1 and not re.search(r"_p\d+$", tid):
                    new_tid = f"{tid}_p{pn}"
                    con.execute(
                        "UPDATE extracted_tables SET table_id = ? WHERE rowid = ?",
                        (new_tid, row_id),
                    )
                    updates += 1
        stats["extracted_tables"] = updates

        # --- ground_truth_diffs: use extracted_tables page mapping ---
        try:
            rows = con.execute(
                "SELECT rowid, table_id FROM ground_truth_diffs "
                "ORDER BY rowid"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        updates = 0
        if rows:
            from collections import Counter
            id_counts = Counter(tid for _, tid in rows)
            for row_id, tid in rows:
                if id_counts[tid] > 1 and not re.search(r"_p\d+$", tid):
                    # Look up page_num from extracted_tables
                    et_row = con.execute(
                        "SELECT page_num FROM extracted_tables "
                        "WHERE table_id = ? LIMIT 1",
                        (tid,),
                    ).fetchone()
                    if et_row:
                        new_tid = f"{tid}_p{et_row[0]}"
                        con.execute(
                            "UPDATE ground_truth_diffs SET table_id = ? WHERE rowid = ?",
                            (new_tid, row_id),
                        )
                        updates += 1
        stats["ground_truth_diffs"] = updates

        # --- paddle_results: group by (item_key, engine_name) ---
        try:
            rows = con.execute(
                "SELECT id, table_id, page_num, item_key, engine_name "
                "FROM paddle_results ORDER BY item_key, engine_name, page_num, id"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        updates = 0
        # Group by (item_key, engine_name)
        groups: dict[tuple[str, str], list[tuple[int, str, int]]] = {}
        for row_id, tid, pn, ik, eng in rows:
            groups.setdefault((ik, eng), []).append((row_id, tid, pn))

        for _key, group in groups.items():
            from collections import Counter
            id_counts = Counter(tid for _, tid, _ in group)
            for row_id, tid, pn in group:
                if id_counts[tid] > 1 and not re.search(r"_p\d+$", tid):
                    new_tid = f"{tid}_p{pn}"
                    con.execute(
                        "UPDATE paddle_results SET table_id = ? WHERE id = ?",
                        (new_tid, row_id),
                    )
                    updates += 1
        stats["paddle_results"] = updates

        # --- paddle_gt_diffs: same logic ---
        try:
            rows = con.execute(
                "SELECT id, table_id, engine_name "
                "FROM paddle_gt_diffs ORDER BY engine_name, id"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        updates = 0
        groups2: dict[str, list[tuple[int, str]]] = {}
        for row_id, tid, eng in rows:
            groups2.setdefault(eng, []).append((row_id, tid))
        for eng, group in groups2.items():
            from collections import Counter
            id_counts = Counter(tid for _, tid in group)
            for row_id, tid in group:
                if id_counts[tid] > 1 and not re.search(r"_p\d+$", tid):
                    # Need page_num from paddle_results
                    pr_row = con.execute(
                        "SELECT page_num FROM paddle_results "
                        "WHERE table_id = ? AND engine_name = ? LIMIT 1",
                        (tid, eng),
                    ).fetchone()
                    if pr_row:
                        new_tid = f"{tid}_p{pr_row[0]}"
                        con.execute(
                            "UPDATE paddle_gt_diffs SET table_id = ? WHERE id = ?",
                            (new_tid, row_id),
                        )
                        updates += 1
        stats["paddle_gt_diffs"] = updates

        # --- Collect base IDs that have collisions (from any already-fixed table) ---
        # A base ID has collisions if any table contains _p\d+ suffixed IDs
        # that were disambiguated.  Used to fix vision tables for the same
        # papers even though vision_run_details PK prevents duplicates.
        colliding_base_ids: set[str] = set()
        for source_table in ("extracted_tables", "paddle_results"):
            try:
                src_rows = con.execute(
                    f"SELECT DISTINCT table_id FROM {source_table} "
                    "WHERE table_id IS NOT NULL"
                ).fetchall()
                for (tid,) in src_rows:
                    m = re.match(r"^(.+_table_[\dA-Z.]+)_p\d+$", tid)
                    if m:
                        colliding_base_ids.add(m.group(1))
            except sqlite3.OperationalError:
                pass

        # --- vision_run_details ---
        # PK prevents duplicates, but IDs need _p{page} for multi-section
        # papers to stay consistent with disambiguated paddle IDs.
        try:
            rows = con.execute(
                "SELECT table_id, page_num FROM vision_run_details "
                "ORDER BY page_num"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        updates = 0
        if rows:
            from collections import Counter
            id_counts = Counter(tid for tid, _ in rows)
            for tid, pn in rows:
                needs_fix = False
                if id_counts[tid] > 1 and not re.search(r"_p\d+$", tid):
                    needs_fix = True  # duplicate within vision_run_details
                elif tid in colliding_base_ids and not re.search(r"_p\d+$", tid):
                    needs_fix = True  # known collision from paddle data
                if needs_fix:
                    new_tid = f"{tid}_p{pn}"
                    con.execute(
                        "UPDATE vision_run_details SET table_id = ? "
                        "WHERE table_id = ? AND page_num = ?",
                        (new_tid, tid, pn),
                    )
                    updates += 1
        stats["vision_run_details"] = updates

        # --- vision_agent_results ---
        # 1. Rename IDs using vision_run_details page mapping
        # 2. Deduplicate: for rows that share (table_id, agent_idx), keep
        #    only the highest-id row (most recently written).
        try:
            rows = con.execute(
                "SELECT var.id, var.table_id, vrd.page_num "
                "FROM vision_agent_results var "
                "LEFT JOIN vision_run_details vrd ON var.table_id = vrd.table_id "
                "ORDER BY var.id"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        updates = 0
        if rows:
            from collections import Counter
            id_counts = Counter(tid for _, tid, _ in rows)
            for row_id, tid, pn in rows:
                needs_fix = False
                if id_counts[tid] > 1 and not re.search(r"_p\d+$", tid):
                    needs_fix = True
                elif tid in colliding_base_ids and not re.search(r"_p\d+$", tid):
                    needs_fix = True
                if needs_fix and pn:
                    new_tid = f"{tid}_p{pn}"
                    con.execute(
                        "UPDATE vision_agent_results SET table_id = ? WHERE id = ?",
                        (new_tid, row_id),
                    )
                    updates += 1

        # Deduplicate: keep highest id per (table_id, agent_idx)
        try:
            dup_ids = con.execute(
                "SELECT id FROM vision_agent_results "
                "WHERE id NOT IN ("
                "  SELECT MAX(id) FROM vision_agent_results "
                "  GROUP BY table_id, agent_idx"
                ")"
            ).fetchall()
            if dup_ids:
                id_list = [r[0] for r in dup_ids]
                placeholders = ",".join("?" * len(id_list))
                con.execute(
                    f"DELETE FROM vision_agent_results WHERE id IN ({placeholders})",
                    id_list,
                )
                updates += len(id_list)
        except sqlite3.OperationalError:
            pass

        stats["vision_agent_results"] = updates

    return stats


def write_paddle_result(db_path: str, result_dict: dict) -> None:
    """Insert one row into the paddle_results table.

    Creates the paddle tables if they do not yet exist on the target connection.
    ``result_dict`` keys correspond directly to the paddle_results column names.
    """
    with sqlite3.connect(db_path) as con:
        con.executescript(PADDLE_SCHEMA)
        con.execute(
            "INSERT INTO paddle_results "
            "(table_id, page_num, engine_name, caption, is_orphan, "
            "headers_json, rows_json, bbox, page_size, raw_output, item_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result_dict.get("table_id"),
                result_dict.get("page_num"),
                result_dict.get("engine_name"),
                result_dict.get("caption"),
                int(bool(result_dict.get("is_orphan", False))),
                result_dict.get("headers_json"),
                result_dict.get("rows_json"),
                result_dict.get("bbox"),
                result_dict.get("page_size"),
                result_dict.get("raw_output"),
                result_dict.get("item_key"),
            ),
        )


def write_paddle_gt_diff(db_path: str, diff_dict: dict) -> None:
    """Insert one row into the paddle_gt_diffs table.

    Creates the paddle tables if they do not yet exist on the target connection.
    ``diff_dict`` keys correspond directly to the paddle_gt_diffs column names.
    """
    with sqlite3.connect(db_path) as con:
        con.executescript(PADDLE_SCHEMA)
        con.execute(
            "INSERT INTO paddle_gt_diffs "
            "(table_id, engine_name, cell_accuracy_pct, fuzzy_accuracy_pct, "
            "num_splits, num_merges, num_cell_diffs, gt_shape, ext_shape, diff_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                diff_dict.get("table_id"),
                diff_dict.get("engine_name"),
                diff_dict.get("cell_accuracy_pct"),
                diff_dict.get("fuzzy_accuracy_pct"),
                diff_dict.get("num_splits"),
                diff_dict.get("num_merges"),
                diff_dict.get("num_cell_diffs"),
                diff_dict.get("gt_shape"),
                diff_dict.get("ext_shape"),
                diff_dict.get("diff_json"),
            ),
        )
