"""Safely reset Axiom's live evaluation data before a fresh testing session.

Deletes every row from the live-evaluation tables (``live_auto_evaluations``,
``live_relevance_labels``, ``live_quality_scores``, ``live_evaluations``,
``live_requests``) in dependency order, inside one transaction. Nothing is deleted
without ``--confirm``.

Untouched by design:

* the SQLite database file, its schema and indexes;
* the legacy ``rag_audit`` application audit log;
* LangGraph conversation memory (``data/langgraph_memory.sqlite``);
* uploaded documents, Pinecone vectors and ``evaluation/dataset.json``;
* offline benchmark results in ``evaluation/results/``.

Usage (from the project root)::

    python reset_live_eval.py --confirm
    python reset_live_eval.py --confirm --database data/audit.db

Without ``--confirm`` the script explains what it would do and exits without
touching the database.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.config import get_settings
from src.live_db import (
    LIVE_EVAL_TABLES,
    ResetNotConfirmed,
    live_evaluation_counts,
    reset_live_evaluation_data,
)


def _default_database() -> Path:
    return get_settings().database_file


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reset Axiom live evaluation data (destructive).")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required. Actually delete the live evaluation rows.",
    )
    parser.add_argument(
        "--database",
        default="",
        help="Path to the SQLite database (default: the configured DATABASE_PATH).",
    )
    args = parser.parse_args(argv)

    database = Path(args.database) if args.database else _default_database()
    tables = ", ".join(LIVE_EVAL_TABLES)

    print("=" * 68)
    print("Axiom - Live Evaluation Reset")
    print("=" * 68)
    print(f"Database : {database}")
    print(f"Tables   : {tables}")

    if not args.confirm:
        print()
        print("Refusing to delete anything: --confirm was not provided.")
        print("This would delete ALL live evaluation rows (quality scores,")
        print("evaluations and requests). Audit logs, LangGraph memory, uploads,")
        print("Pinecone vectors, dataset.json and evaluation/results/ are kept.")
        print()
        print("Re-run with --confirm to proceed:")
        print("    python reset_live_eval.py --confirm")
        return 2

    try:
        deleted = reset_live_evaluation_data(confirm=True, path=database)
    except ResetNotConfirmed as exc:  # pragma: no cover - guarded by --confirm
        print(f"ERROR: {exc}")
        return 2

    print()
    print("Deleted rows per table:")
    for table in LIVE_EVAL_TABLES:
        print(f"  {table:<24} {deleted.get(table, 0)}")

    remaining = live_evaluation_counts(path=database)
    print()
    print("Remaining rows per table:")
    for table in LIVE_EVAL_TABLES:
        print(f"  {table:<24} {remaining.get(table, 0)}")

    if any(remaining.get(table, 0) for table in LIVE_EVAL_TABLES):
        print()
        print("WARNING: some live evaluation rows remain.")
        return 1

    print()
    print("Live evaluation data reset complete (database file and schema kept).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
