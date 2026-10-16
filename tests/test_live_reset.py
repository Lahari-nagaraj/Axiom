"""Offline tests for the live evaluation reset procedure.

Uses a temporary SQLite database only - the real ``data/audit.db`` is never
touched. No network/LLM calls and no benchmark runs.
"""

import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import reset_live_eval
from src import live_db


class LiveResetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db = self.tmp / "audit.db"
        live_db.init_live_db(self.db)
        self._seed()

    def tearDown(self):
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def _exec(self, sql, params=()):
        conn = sqlite3.connect(self.db)
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _seed(self):
        for i in range(2):
            self._exec(
                "INSERT INTO live_requests (request_id, created_at, question, status) VALUES (?, ?, ?, ?)",
                (f"r{i}", f"2026-10-0{i + 1}T10:00:00+00:00", f"q{i}", "ok"),
            )
        for rid in ("r0", "r1"):
            for system in ("axiom_self_rag", "baseline_rag"):
                self._exec(
                    "INSERT INTO live_evaluations (request_id, system_version, created_at, status) "
                    "VALUES (?, ?, ?, ?)",
                    (rid, system, "2026-10-01T10:00:00+00:00", "ok"),
                )
        self._exec(
            "INSERT INTO live_quality_scores (request_id, system_version, created_at, method) "
            "VALUES (?, ?, ?, ?)",
            ("r0", "axiom_self_rag", "2026-10-01T10:00:00+00:00", "manual"),
        )
        self._exec(
            "INSERT INTO live_auto_evaluations (request_id, system_version, created_at, method, status, metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("r0", "axiom_self_rag", "2026-10-01T10:00:00+00:00", "lexical_proxy_v1", "ok", "{}"),
        )
        for rid in ("r0", "r1"):
            self._exec(
                "INSERT INTO live_relevance_labels (request_id, question, source_id, relevant, labeled_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (rid, f"q{rid[-1]}", f"doc-{rid}.md", 1, "2026-10-01T10:00:00+00:00"),
            )
        # Unrelated tables that must survive the reset.
        self._exec(
            "CREATE TABLE IF NOT EXISTS rag_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT)"
        )
        self._exec("INSERT INTO rag_audit (question, answer) VALUES (?, ?)", ("kept?", "yes"))
        self._exec("CREATE TABLE IF NOT EXISTS memory (thread_id TEXT, entry TEXT)")
        self._exec("INSERT INTO memory (thread_id, entry) VALUES (?, ?)", ("thread-1", "User: hi"))

    def _count(self, table):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()

    def _object_names(self):
        conn = sqlite3.connect(self.db)
        try:
            return {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        finally:
            conn.close()

    # -- tests -------------------------------------------------------------
    def test_requires_confirmation_and_deletes_nothing(self):
        with self.assertRaises(live_db.ResetNotConfirmed):
            live_db.reset_live_evaluation_data(path=self.db)  # confirm defaults to False
        self.assertEqual(self._count("live_requests"), 2)
        self.assertEqual(self._count("live_evaluations"), 4)
        self.assertEqual(self._count("live_quality_scores"), 1)
        self.assertEqual(self._count("live_relevance_labels"), 2)
        self.assertEqual(self._count("live_auto_evaluations"), 1)

    def test_deletes_all_tables_and_reports_counts(self):
        deleted = live_db.reset_live_evaluation_data(confirm=True, path=self.db)
        self.assertEqual(
            deleted,
            {
                "live_auto_evaluations": 1,
                "live_relevance_labels": 2,
                "live_quality_scores": 1,
                "live_evaluations": 4,
                "live_requests": 2,
            },
        )
        for table in live_db.LIVE_EVAL_TABLES:
            self.assertEqual(self._count(table), 0, table)

    def test_deletion_order_is_children_first(self):
        self.assertEqual(
            live_db.LIVE_EVAL_TABLES,
            (
                "live_auto_evaluations",
                "live_relevance_labels",
                "live_quality_scores",
                "live_evaluations",
                "live_requests",
            ),
        )

    def test_preserves_schema_and_indexes(self):
        before = self._object_names()
        live_db.reset_live_evaluation_data(confirm=True, path=self.db)
        self.assertEqual(self._object_names(), before)
        for index in (
            "idx_live_requests_created",
            "idx_live_evaluations_request",
            "idx_live_evaluations_created",
            "idx_live_evaluations_system",
            "idx_live_quality_request",
            "idx_live_auto_request",
            "idx_live_labels_request",
            "idx_live_labels_source",
        ):
            self.assertIn(index, before)

    def test_preserves_unrelated_tables(self):
        live_db.reset_live_evaluation_data(confirm=True, path=self.db)
        self.assertEqual(self._count("rag_audit"), 1)
        self.assertEqual(self._count("memory"), 1)

    def test_rolls_back_on_failure(self):
        # Block deletion of the parent table so the transaction must abort.
        self._exec(
            "CREATE TRIGGER block_request_delete BEFORE DELETE ON live_requests "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END;"
        )
        with self.assertRaises(sqlite3.Error):
            live_db.reset_live_evaluation_data(confirm=True, path=self.db)
        # The earlier DELETEs in the same transaction are rolled back.
        self.assertEqual(self._count("live_quality_scores"), 1)
        self.assertEqual(self._count("live_evaluations"), 4)
        self.assertEqual(self._count("live_requests"), 2)

    def test_counts_helper(self):
        counts = live_db.live_evaluation_counts(path=self.db)
        self.assertEqual(
            counts,
            {
                "live_auto_evaluations": 1,
                "live_relevance_labels": 2,
                "live_quality_scores": 1,
                "live_evaluations": 4,
                "live_requests": 2,
            },
        )

    def test_reset_on_fresh_database_is_idempotent(self):
        live_db.reset_live_evaluation_data(confirm=True, path=self.db)
        again = live_db.reset_live_evaluation_data(confirm=True, path=self.db)
        self.assertEqual(
            again,
            {
                "live_auto_evaluations": 0,
                "live_relevance_labels": 0,
                "live_quality_scores": 0,
                "live_evaluations": 0,
                "live_requests": 0,
            },
        )


class ResetCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db = self.tmp / "audit.db"
        live_db.init_live_db(self.db)
        conn = sqlite3.connect(self.db)
        try:
            conn.execute(
                "INSERT INTO live_requests (request_id, created_at, question, status) VALUES (?, ?, ?, ?)",
                ("r0", "2026-10-01T10:00:00+00:00", "q0", "ok"),
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self._tmp.cleanup()

    def _count(self):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute("SELECT COUNT(*) FROM live_requests").fetchone()[0]
        finally:
            conn.close()

    def test_without_confirm_exits_2_and_keeps_data(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = reset_live_eval.main(["--database", str(self.db)])
        self.assertEqual(rc, 2)
        self.assertEqual(self._count(), 1)
        self.assertIn("--confirm was not provided", out.getvalue())

    def test_with_confirm_resets_and_prints_counts(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = reset_live_eval.main(["--confirm", "--database", str(self.db)])
        self.assertEqual(rc, 0)
        self.assertEqual(self._count(), 0)
        text = out.getvalue()
        self.assertIn("live_quality_scores", text)
        self.assertIn("live_evaluations", text)
        self.assertIn("live_requests", text)
        self.assertIn("reset complete", text)


if __name__ == "__main__":
    unittest.main()
