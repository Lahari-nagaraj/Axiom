"""Durable storage for human-scored evaluation columns.

Manual scores are stored in a small CSV keyed by
``(run_id, system_version, question_id)`` so that regenerating a run never
silently loses reviewed scores.

The per-question CSV columns (``answer_correctness_manual``,
``faithfulness_manual``, ``manual_notes``) remain the backwards-compatible place
a reviewer edits. This file is a durable overlay: it is merged into the
per-question rows before aggregation, and updated whenever non-empty manual
scores are seen, so a fresh result file with blank manual columns can never
erase scores that already exist for the same key.

Nothing here fabricates a score: blank cells are preserved as blank.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

MANUAL_FIELDS: Tuple[str, ...] = (
    "answer_correctness_manual",
    "faithfulness_manual",
    "manual_notes",
)

SCORE_COLUMNS: List[str] = [
    "run_id",
    "system_version",
    "question_id",
    *MANUAL_FIELDS,
    "updated_at",
]

MANUAL_SCORES_FILENAME = "manual_scores.csv"

Key = Tuple[str, str, str]


def _key(row: Dict[str, Any]) -> Key:
    return (
        str(row.get("run_id", "") or ""),
        str(row.get("system_version", "") or ""),
        str(row.get("question_id", "") or ""),
    )


def _norm(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _has_score(row: Dict[str, Any]) -> bool:
    return any(_norm(row.get(field)) for field in MANUAL_FIELDS)


def _clean(row: Dict[str, Any]) -> Dict[str, str]:
    return {field: _norm(row.get(field)) for field in MANUAL_FIELDS}


def load_manual_scores(path: Path) -> Dict[Key, Dict[str, str]]:
    """Read the durable score store. Missing file -> empty store."""
    path = Path(path)
    if not path.exists():
        return {}
    store: Dict[Key, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row:
                continue
            store[_key(row)] = _clean(row)
    return store


def collect_manual_scores(rows: Iterable[Dict[str, Any]]) -> Dict[Key, Dict[str, str]]:
    """Extract the non-empty manual scores already present on ``rows``."""
    out: Dict[Key, Dict[str, str]] = {}
    for row in rows:
        if _has_score(row):
            out[_key(row)] = _clean(row)
    return out


def apply_manual_scores(rows: Iterable[Dict[str, Any]], store: Dict[Key, Dict[str, str]]) -> int:
    """Fill blank manual cells from ``store``. Existing row values win.

    Returns the number of rows that received at least one stored value.
    """
    filled = 0
    for row in rows:
        if _has_score(row):
            continue
        saved = store.get(_key(row))
        if not saved or not any(_norm(v) for v in saved.values()):
            continue
        for field in MANUAL_FIELDS:
            row[field] = saved.get(field, "")
        filled += 1
    return filled


def persist_manual_scores(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Merge non-empty manual scores from ``rows`` into the durable store.

    Existing entries for keys not present in ``rows`` are kept, so scores from
    other runs or earlier files are never lost. Returns the number of stored
    keys, or 0 if there is nothing to store (the file is not created).
    """
    path = Path(path)
    store = load_manual_scores(path)
    for key, vals in collect_manual_scores(rows).items():
        store[key] = vals
    if not store:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for key in sorted(store):
            vals = store[key]
            writer.writerow(
                {
                    "run_id": key[0],
                    "system_version": key[1],
                    "question_id": key[2],
                    **{field: vals.get(field, "") for field in MANUAL_FIELDS},
                    "updated_at": now,
                }
            )
    return len(store)
