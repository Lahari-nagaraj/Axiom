"""Persistent storage for live (runtime) evaluation records.

This reuses the existing SQLite database configured by ``src.config`` (the same
file as the ``rag_audit`` table) and only *adds* tables. It is the durable source
of truth for the live dashboard: request records, per-system evaluation results,
and any available quality scores. Nothing here writes to CSV or memory-only
structures.

Tables
------
``live_requests``
    One row per real frontend question. Holds the shared raw retrieval (used by
    retrieval metrics), the Self-RAG response fields, latency, status, error and
    token/cost metadata.
``live_evaluations``
    One row per ``(request_id, system_version)``: the per-system answer, latency,
    sources, routing/verification outcomes and usage. ``UNIQUE`` prevents a
    system from being evaluated twice for the same request.
``live_quality_scores``
    Optional, explicitly separated quality judgements (a manual review or the
    optional LLM judge). Absent rows mean "no judgement available" - never 0.
``live_auto_evaluations``
    The persisted per-question automatic evaluation computed locally from the
    recorded execution artifacts (question, retrieved chunk text/ranks/scores,
    final evidence, answer). Keyed ``UNIQUE(request_id, system_version, method)``
    so opening a detail view never creates a duplicate evaluation and never
    re-runs answer generation.
``live_relevance_labels``
    Legacy, explicit (non-LLM) relevance judgments. Retained so previously stored
    data is never lost; the live evaluation no longer depends on them. A
    retrieved document is never assumed relevant.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS live_requests (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    thread_id TEXT,
    question TEXT NOT NULL,
    system_version TEXT,
    status TEXT NOT NULL,
    error TEXT,
    error_category TEXT,
    latency_seconds REAL,
    answer TEXT,
    route TEXT,
    used_web_search INTEGER DEFAULT 0,
    support_status TEXT,
    usefulness TEXT,
    retries INTEGER,
    retrieval_available INTEGER DEFAULT 0,
    retrieved_sources_json TEXT,
    retrieved_chunks_json TEXT,
    sources_json TEXT,
    trace_json TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    token_usage_available INTEGER DEFAULT 0,
    estimated_cost_usd REAL
);

CREATE TABLE IF NOT EXISTS live_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    system_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    error_category TEXT,
    latency_seconds REAL,
    answer TEXT,
    route TEXT,
    retries INTEGER,
    support_status TEXT,
    usefulness TEXT,
    used_web_search INTEGER DEFAULT 0,
    retrieval_available INTEGER DEFAULT 0,
    retrieved_sources_json TEXT,
    retrieved_chunks_json TEXT,
    used_sources_json TEXT,
    final_evidence_json TEXT,
    grades_json TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    token_usage_available INTEGER DEFAULT 0,
    estimated_cost_usd REAL,
    UNIQUE (request_id, system_version)
);

CREATE TABLE IF NOT EXISTS live_quality_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    system_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    method TEXT NOT NULL,
    answer_correctness REAL,
    faithfulness REAL,
    citation_precision REAL,
    citation_recall REAL,
    notes TEXT,
    UNIQUE (request_id, system_version, method)
);

-- Persisted per-question automatic evaluation. Keyed by
-- (request_id, system_version, method) so it is computed once and reused; the
-- dashboard never re-runs answer generation to open a question.
CREATE TABLE IF NOT EXISTS live_auto_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    system_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    method TEXT NOT NULL,
    status TEXT,
    metrics_json TEXT NOT NULL,
    UNIQUE (request_id, system_version, method)
);

-- Legacy, explicit (non-LLM) relevance judgments. Retained for data
-- preservation; the live evaluation no longer depends on them. ``source_id`` is a
-- normalized source/document identifier - a retrieved document is NEVER assumed
-- relevant - a row only exists when someone explicitly labeled it.
CREATE TABLE IF NOT EXISTS live_relevance_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    question TEXT,
    source_id TEXT NOT NULL,
    relevant INTEGER NOT NULL,
    labeled_by TEXT,
    labeled_at TEXT NOT NULL,
    notes TEXT,
    UNIQUE (request_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_live_requests_created ON live_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_live_evaluations_request ON live_evaluations (request_id);
CREATE INDEX IF NOT EXISTS idx_live_evaluations_created ON live_evaluations (created_at);
CREATE INDEX IF NOT EXISTS idx_live_evaluations_system ON live_evaluations (system_version);
CREATE INDEX IF NOT EXISTS idx_live_quality_request ON live_quality_scores (request_id);
CREATE INDEX IF NOT EXISTS idx_live_auto_request ON live_auto_evaluations (request_id);
CREATE INDEX IF NOT EXISTS idx_live_labels_request ON live_relevance_labels (request_id);
CREATE INDEX IF NOT EXISTS idx_live_labels_source ON live_relevance_labels (source_id);
"""

# Columns added after the first release of a table. ``CREATE TABLE IF NOT EXISTS``
# never alters an existing table, so ``init_live_db`` applies these additively
# (existing rows keep their values; missing columns become NULL).
ADDED_COLUMNS = {
    "live_requests": {
        "retrieved_chunks_json": "TEXT",
    },
    "live_evaluations": {
        "retrieved_chunks_json": "TEXT",
        "final_evidence_json": "TEXT",
        "grades_json": "TEXT",
    },
}


REQUEST_FIELDS = (
    "request_id",
    "created_at",
    "thread_id",
    "question",
    "system_version",
    "status",
    "error",
    "error_category",
    "latency_seconds",
    "answer",
    "route",
    "used_web_search",
    "support_status",
    "usefulness",
    "retries",
    "retrieval_available",
    "retrieved_sources_json",
    "retrieved_chunks_json",
    "sources_json",
    "trace_json",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_usage_available",
    "estimated_cost_usd",
)

EVALUATION_FIELDS = (
    "request_id",
    "system_version",
    "created_at",
    "status",
    "error",
    "error_category",
    "latency_seconds",
    "answer",
    "route",
    "retries",
    "support_status",
    "usefulness",
    "used_web_search",
    "retrieval_available",
    "retrieved_sources_json",
    "retrieved_chunks_json",
    "used_sources_json",
    "final_evidence_json",
    "grades_json",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_usage_available",
    "estimated_cost_usd",
)

QUALITY_FIELDS = (
    "request_id",
    "system_version",
    "created_at",
    "method",
    "answer_correctness",
    "faithfulness",
    "citation_precision",
    "citation_recall",
    "notes",
)

RELEVANCE_LABEL_FIELDS = (
    "request_id",
    "question",
    "source_id",
    "relevant",
    "labeled_by",
    "labeled_at",
    "notes",
)

AUTO_EVAL_FIELDS = (
    "request_id",
    "system_version",
    "created_at",
    "method",
    "status",
    "metrics_json",
)

# The live-evaluation tables, in deletion (dependency) order: children first.
# There are no SQL ``FOREIGN KEY`` constraints, but every non-parent table
# references ``live_requests.request_id``, so they are cleared before the parent.
LIVE_EVAL_TABLES = (
    "live_auto_evaluations",
    "live_relevance_labels",
    "live_quality_scores",
    "live_evaluations",
    "live_requests",
)


class ResetNotConfirmed(RuntimeError):
    """Raised when a destructive live-evaluation reset is attempted without confirmation."""


# --------------------------------------------------------------------------- #
# Source/document identifier normalization
# --------------------------------------------------------------------------- #
def normalize_source_id(value: Any) -> str:
    """Normalize a source/document identifier for consistent comparison.

    Retrieved sources are stored as document names, but a label may be entered as
    a full path, with different separators or letter case. Normalization removes
    the directory part, normalizes separators and lower-cases, so the same
    document matches regardless of how it was written. Empty input -> ``""``.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    text = text.replace("\\", "/").rstrip("/")
    base = text.rsplit("/", 1)[-1] if "/" in text else text
    return base.strip().lower()


def _resolve_path(path: Optional[Any] = None) -> Path:
    if path is not None:
        return Path(path)
    return get_settings().database_file


@contextmanager
def _connect(path: Optional[Any] = None):
    db_path = _resolve_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_live_db(path: Optional[Any] = None) -> None:
    """Create the live-evaluation tables/indexes if they do not exist.

    Additive only: existing tables are never dropped or rewritten. Any column
    introduced after a table's first release is added with ``ALTER TABLE ADD
    COLUMN`` (existing rows keep their values; the new column is NULL).
    """
    with _connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA_SQL)
        for table, columns in ADDED_COLUMNS.items():
            _ensure_columns(conn, table, columns)


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if not existing:
        return
    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _insert(conn: sqlite3.Connection, table: str, fields, record: Dict[str, Any], *, ignore: bool) -> None:
    columns = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    sql = (
        f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})"
        if ignore
        else f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})"
    )
    conn.execute(sql, [record.get(field) for field in fields])


def save_request(record: Dict[str, Any], path: Optional[Any] = None) -> None:
    """Insert or replace a request record (keyed by ``request_id``)."""
    with _connect(path) as conn:
        _insert(conn, "live_requests", REQUEST_FIELDS, record, ignore=False)


def save_evaluation(record: Dict[str, Any], path: Optional[Any] = None) -> bool:
    """Insert a per-system evaluation once. Returns True if a new row was added.

    The ``UNIQUE(request_id, system_version)`` constraint makes this idempotent so
    the same system can never be executed/recorded twice for one request.
    """
    with _connect(path) as conn:
        before = conn.total_changes
        _insert(conn, "live_evaluations", EVALUATION_FIELDS, record, ignore=True)
        return conn.total_changes > before


def has_evaluation(request_id: str, system_version: str, path: Optional[Any] = None) -> bool:
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM live_evaluations WHERE request_id = ? AND system_version = ? LIMIT 1",
            (request_id, system_version),
        ).fetchone()
    return row is not None


def set_request_retrieval(
    request_id: str,
    retrieved_sources: List[str],
    retrieval_available: bool,
    path: Optional[Any] = None,
    *,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Attach the shared raw retrieval to a request (idempotent update).

    ``retrieved_chunks`` (rank/source/title/text/score) is stored when supplied;
    a ``None`` value leaves any previously stored chunk artifact untouched.
    """
    with _connect(path) as conn:
        conn.execute(
            "UPDATE live_requests SET retrieved_sources_json = ?, retrieval_available = ?, "
            "retrieved_chunks_json = COALESCE(?, retrieved_chunks_json) WHERE request_id = ?",
            (
                json.dumps(retrieved_sources or [], ensure_ascii=False),
                int(bool(retrieval_available)),
                json.dumps(retrieved_chunks, ensure_ascii=False) if retrieved_chunks is not None else None,
                request_id,
            ),
        )


def save_quality_score(record: Dict[str, Any], path: Optional[Any] = None) -> bool:
    with _connect(path) as conn:
        before = conn.total_changes
        _insert(conn, "live_quality_scores", QUALITY_FIELDS, record, ignore=True)
        return conn.total_changes > before


def _decode_json(value: Any, default: Any):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _request_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    data["used_web_search"] = bool(data.get("used_web_search"))
    data["retrieval_available"] = bool(data.get("retrieval_available"))
    data["token_usage_available"] = bool(data.get("token_usage_available"))
    data["retrieved_sources"] = _decode_json(data.pop("retrieved_sources_json", None), [])
    data["retrieved_chunks"] = _decode_json(data.pop("retrieved_chunks_json", None), [])
    data["sources"] = _decode_json(data.pop("sources_json", None), [])
    data["trace"] = _decode_json(data.pop("trace_json", None), [])
    return data


def _evaluation_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    data["used_web_search"] = bool(data.get("used_web_search"))
    data["retrieval_available"] = bool(data.get("retrieval_available"))
    data["token_usage_available"] = bool(data.get("token_usage_available"))
    data["retrieved_sources"] = _decode_json(data.pop("retrieved_sources_json", None), [])
    data["retrieved_chunks"] = _decode_json(data.pop("retrieved_chunks_json", None), [])
    data["used_sources"] = _decode_json(data.pop("used_sources_json", None), [])
    data["final_evidence"] = _decode_json(data.pop("final_evidence_json", None), [])
    data["grades"] = _decode_json(data.pop("grades_json", None), [])
    return data


def _date_clause(date_from: Optional[str], date_to: Optional[str], clauses: List[str], params: List[Any]) -> None:
    if date_from:
        clauses.append("date(created_at) >= date(?)")
        params.append(date_from)
    if date_to:
        clauses.append("date(created_at) <= date(?)")
        params.append(date_to)


def get_request(request_id: str, path: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM live_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
    return _request_to_dict(row) if row else None


def list_requests(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    system: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    path: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    _date_clause(date_from, date_to, clauses, params)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if system:
        clauses.append(
            "EXISTS (SELECT 1 FROM live_evaluations e WHERE e.request_id = live_requests.request_id AND e.system_version = ?)"
        )
        params.append(system)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    order = " ORDER BY created_at DESC, request_id DESC"
    sql = f"SELECT * FROM live_requests{where}{order}"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
    with _connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_request_to_dict(r) for r in rows]


def count_requests(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    system: Optional[str] = None,
    path: Optional[Any] = None,
) -> int:
    clauses: List[str] = []
    params: List[Any] = []
    _date_clause(date_from, date_to, clauses, params)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if system:
        clauses.append(
            "EXISTS (SELECT 1 FROM live_evaluations e WHERE e.request_id = live_requests.request_id AND e.system_version = ?)"
        )
        params.append(system)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM live_requests{where}", params).fetchone()
    return int(row["n"]) if row else 0


def list_evaluations(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    system: Optional[str] = None,
    request_id: Optional[str] = None,
    path: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    _date_clause(date_from, date_to, clauses, params)
    if system:
        clauses.append("system_version = ?")
        params.append(system)
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as conn:
        rows = conn.execute(
            f"SELECT * FROM live_evaluations{where} ORDER BY created_at DESC, id DESC", params
        ).fetchall()
    return [_evaluation_to_dict(r) for r in rows]


def list_quality_scores(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    path: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    _date_clause(date_from, date_to, clauses, params)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as conn:
        rows = conn.execute(
            f"SELECT * FROM live_quality_scores{where} ORDER BY created_at DESC, id DESC", params
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Persisted per-question automatic evaluations
# --------------------------------------------------------------------------- #
def _auto_eval_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    data["metrics"] = _decode_json(data.pop("metrics_json", None), {})
    return data


def save_auto_evaluation(record: Dict[str, Any], path: Optional[Any] = None) -> None:
    """Insert or update one automatic evaluation, keyed by request/system/method.

    Idempotent: recomputing the same question does not create a second row, so a
    detail-view refresh can never duplicate evaluations.
    """
    metrics = record.get("metrics")
    if metrics is None:
        metrics = {}
    payload = {
        "request_id": record.get("request_id"),
        "system_version": record.get("system_version"),
        "created_at": record.get("created_at") or _utc_now(),
        "method": record.get("method"),
        "status": record.get("status"),
        "metrics_json": metrics if isinstance(metrics, str) else json.dumps(metrics, ensure_ascii=False),
    }
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO live_auto_evaluations
                (request_id, system_version, created_at, method, status, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id, system_version, method) DO UPDATE SET
                created_at = excluded.created_at,
                status = excluded.status,
                metrics_json = excluded.metrics_json
            """,
            [payload[field] for field in AUTO_EVAL_FIELDS],
        )


def get_auto_evaluation(
    request_id: str,
    system_version: str,
    method: Optional[str] = None,
    path: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    clauses = ["request_id = ?", "system_version = ?"]
    params: List[Any] = [request_id, system_version]
    if method:
        clauses.append("method = ?")
        params.append(method)
    with _connect(path) as conn:
        row = conn.execute(
            f"SELECT * FROM live_auto_evaluations WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT 1",
            params,
        ).fetchone()
    return _auto_eval_to_dict(row) if row else None


def list_auto_evaluations(
    request_id: Optional[str] = None,
    path: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as conn:
        rows = conn.execute(
            f"SELECT * FROM live_auto_evaluations{where} ORDER BY request_id, system_version, method", params
        ).fetchall()
    return [_auto_eval_to_dict(r) for r in rows]


def _relevance_label_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    data["relevant"] = bool(data.get("relevant"))
    return data


def save_relevance_label(record: Dict[str, Any], path: Optional[Any] = None) -> Dict[str, Any]:
    """Insert or update a relevance judgment (keyed by ``request_id`` + ``source_id``).

    ``source_id`` is normalized with :func:`normalize_source_id` before storage so
    judgments and retrieved documents match regardless of path/case differences.
    Saving the same question/source pair again edits the existing judgment
    (relevant flag, labeler, notes and timestamp) instead of duplicating it.
    """
    normalized = normalize_source_id(record.get("source_id"))
    if not normalized:
        raise ValueError("source_id must be a non-empty source/document identifier")

    payload = {
        "request_id": record.get("request_id"),
        "question": record.get("question"),
        "source_id": normalized,
        "relevant": int(bool(record.get("relevant"))),
        "labeled_by": record.get("labeled_by"),
        "labeled_at": record.get("labeled_at") or _utc_now(),
        "notes": record.get("notes"),
    }
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO live_relevance_labels
                (request_id, question, source_id, relevant, labeled_by, labeled_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id, source_id) DO UPDATE SET
                question = excluded.question,
                relevant = excluded.relevant,
                labeled_by = excluded.labeled_by,
                labeled_at = excluded.labeled_at,
                notes = excluded.notes
            """,
            [payload[field] for field in RELEVANCE_LABEL_FIELDS],
        )
        row = conn.execute(
            "SELECT * FROM live_relevance_labels WHERE request_id = ? AND source_id = ?",
            (payload["request_id"], normalized),
        ).fetchone()
    return _relevance_label_to_dict(row) if row else payload


def delete_relevance_label(request_id: str, source_id: str, path: Optional[Any] = None) -> bool:
    """Remove one judgment. Returns True when a row was deleted."""
    normalized = normalize_source_id(source_id)
    with _connect(path) as conn:
        before = conn.total_changes
        conn.execute(
            "DELETE FROM live_relevance_labels WHERE request_id = ? AND source_id = ?",
            (request_id, normalized),
        )
        return conn.total_changes > before


def list_relevance_labels(
    request_id: Optional[str] = None,
    path: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """All stored judgments, optionally for a single request."""
    clauses: List[str] = []
    params: List[Any] = []
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as conn:
        rows = conn.execute(
            f"SELECT * FROM live_relevance_labels{where} ORDER BY request_id, source_id", params
        ).fetchall()
    return [_relevance_label_to_dict(r) for r in rows]


def count_relevance_labels(request_id: Optional[str] = None, path: Optional[Any] = None) -> int:
    clauses: List[str] = []
    params: List[Any] = []
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM live_relevance_labels{where}", params
        ).fetchone()
    return int(row["n"]) if row else 0


def count_evaluated_requests(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    path: Optional[Any] = None,
) -> int:
    clauses: List[str] = []
    params: List[Any] = []
    _date_clause(date_from, date_to, clauses, params)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as conn:
        row = conn.execute(
            f"SELECT COUNT(DISTINCT request_id) AS n FROM live_evaluations{where}", params
        ).fetchone()
    return int(row["n"]) if row else 0


def live_evaluation_counts(path: Optional[Any] = None) -> Dict[str, int]:
    """Row counts for each live-evaluation table (0 when a table is missing)."""
    counts: Dict[str, int] = {}
    with _connect(path) as conn:
        existing = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        for table in LIVE_EVAL_TABLES:
            if table not in existing:
                counts[table] = 0
                continue
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[table] = int(row["n"]) if row else 0
    return counts


def reset_live_evaluation_data(*, confirm: bool = False, path: Optional[Any] = None) -> Dict[str, int]:
    """Delete every row from the live-evaluation tables in dependency order.

    Safe reset for a fresh testing session:

    * Requires ``confirm=True`` - without it, nothing is touched and
      :class:`ResetNotConfirmed` is raised.
    * Deletes children before parents (``live_auto_evaluations`` ->
      ``live_relevance_labels`` -> ``live_quality_scores`` -> ``live_evaluations``
      -> ``live_requests``) inside a **single transaction**, so the deletion
      either completes fully or rolls back completely.
    * Only these tables are modified. The database file, schema, indexes,
      the legacy ``rag_audit`` log, LangGraph conversation memory, uploads,
      Pinecone vectors and ``evaluation/dataset.json`` are left untouched.

    Returns a mapping of ``table -> number of deleted rows``.
    """
    if not confirm:
        raise ResetNotConfirmed(
            "Refusing to delete live evaluation data without confirmation "
            "(pass confirm=True)."
        )

    # Ensure the tables exist so the reset works on a fresh database. This only
    # (re)creates the identical schema and never touches existing rows.
    init_live_db(path)

    deleted: Dict[str, int] = {}
    with _connect(path) as conn:
        for table in LIVE_EVAL_TABLES:
            cursor = conn.execute(f"DELETE FROM {table}")
            deleted[table] = max(0, cursor.rowcount)
    return deleted
