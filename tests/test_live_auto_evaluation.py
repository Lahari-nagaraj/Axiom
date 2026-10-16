"""Offline tests for the automatic, question-level live evaluation.

Covers: real retrieved chunk artifacts (source, rank, text) drive the automatic
metrics; Baseline RAG and Self-RAG are evaluated separately; initial retrieval and
final Self-RAG evidence are never conflated; no manual relevance labels are needed;
missing data and missing chunk text are reported honestly; conventional retrieval
metrics are unavailable rather than fabricated; the stored evaluation is not
duplicated; opening a detail view never regenerates the answer; and the optional
judge stays off unless explicitly configured.

No network/LLM calls: the database is a per-test temporary SQLite file.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import auto_eval, live_db, live_eval, live_metrics

RID = "req-1"
TS = "2026-10-01T10:00:00+00:00"
CHUNK_1 = {
    "rank": 1,
    "source": "checkout-api-runbook.md",
    "title": "Checkout API Runbook",
    "type": "internal",
    "page": 2,
    "text": "checkout api returns 502 after deployment; compare incident start with deployment time",
}
CHUNK_2 = {
    "rank": 2,
    "source": "db-runbook.md",
    "title": "DB Runbook",
    "type": "internal",
    "page": None,
    "text": "database connection pool exhausted under load",
}


class FakeSettings:
    def __init__(self, path, *, judge_enabled=False):
        self.database_path = str(path)
        self.database_file = Path(path)
        self.live_eval_enabled = True
        self.live_eval_baseline_enabled = True
        self.live_eval_max_concurrency = 1
        self.live_eval_judge_enabled = judge_enabled
        self.groq_input_cost_per_1m = None
        self.groq_output_cost_per_1m = None


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db = self.tmp / "audit.db"
        live_db.init_live_db(self.db)
        live_eval.reset_runtime_state()

    def tearDown(self):
        live_eval.reset_runtime_state()
        self._tmp.cleanup()

    def seed(self, *, baseline=True, self_rag=True, chunks=(CHUNK_1, CHUNK_2), final=(CHUNK_1,)):
        live_db.save_request(
            live_eval._request_record(
                RID, TS, "How do I fix the checkout api 502 errors after deployment?",
                "thread-1", status="ok", latency_seconds=1.0, result=None,
            ),
            self.db,
        )
        live_db.set_request_retrieval(
            RID, ["checkout-api-runbook.md", "db-runbook.md"], True, self.db, retrieved_chunks=list(chunks)
        )
        if baseline:
            live_db.save_evaluation(
                live_eval._evaluation_record(
                    RID, live_eval.BASELINE_SYSTEM, TS, status="ok", latency_seconds=1.1,
                    answer="Restart the checkout api deployment.",
                    route="internal_retrieval",
                    retrieval_available=True,
                    retrieved_sources=["checkout-api-runbook.md", "db-runbook.md"],
                    retrieved_chunks=list(chunks),
                    used_sources=["checkout-api-runbook.md", "db-runbook.md"],
                ),
                self.db,
            )
        if self_rag:
            live_db.save_evaluation(
                live_eval._evaluation_record(
                    RID, live_eval.SELF_RAG_SYSTEM, TS, status="ok", latency_seconds=0.9,
                    answer="Check the deployment time for the checkout api.",
                    route="Private Runbooks",
                    support_status="fully_supported",
                    usefulness="useful",
                    retries=0,
                    retrieval_available=True,
                    retrieved_sources=["checkout-api-runbook.md", "db-runbook.md"],
                    retrieved_chunks=list(chunks),
                    used_sources=["checkout-api-runbook.md"],
                    final_evidence=list(final),
                    grades=[
                        {"source": "checkout-api-runbook.md", "type": "internal", "relevant": True},
                        {"source": "db-runbook.md", "type": "internal", "relevant": False},
                    ],
                ),
                self.db,
            )


class DetailTests(_Base):
    def test_real_chunks_and_separate_systems(self):
        self.seed()
        detail = live_metrics.build_request_detail(RID, self.db)
        baseline = detail["systems"][live_eval.BASELINE_SYSTEM]
        self_rag = detail["systems"][live_eval.SELF_RAG_SYSTEM]

        self.assertEqual(baseline["view"], "initial_retrieval")
        self.assertEqual(self_rag["view"], "final_evidence")

        # Actual chunk ids, ranks and text come from the stored records.
        self.assertEqual([c["rank"] for c in baseline["retrieved_chunks"]], [1, 2])
        self.assertEqual(baseline["retrieved_chunks"][0]["source"], "checkout-api-runbook.md")
        self.assertTrue(baseline["retrieved_chunks"][0]["text"])

        # Self-RAG keeps its initial retrieval and its final evidence distinct.
        self.assertEqual(len(self_rag["retrieved_chunks"]), 2)
        self.assertEqual(len(self_rag["final_evidence"]), 1)
        self.assertEqual(self_rag["final_evidence"][0]["source"], "checkout-api-runbook.md")

    def test_automatic_metrics_computed_per_system(self):
        self.seed()
        detail = live_metrics.build_request_detail(RID, self.db)
        baseline = detail["systems"][live_eval.BASELINE_SYSTEM]["automatic"]
        self_rag = detail["systems"][live_eval.SELF_RAG_SYSTEM]["automatic"]
        self.assertEqual(baseline["status"], "ok")
        self.assertEqual(self_rag["status"], "ok")
        self.assertIsNotNone(baseline["mean_query_term_coverage"])
        self.assertIsNotNone(self_rag["top_rank_query_term_coverage"])

    def test_answers_and_verification_surfaced(self):
        self.seed()
        detail = live_metrics.build_request_detail(RID, self.db)
        self.assertIn("checkout", detail["systems"][live_eval.BASELINE_SYSTEM]["answer"].lower())
        ver = detail["systems"][live_eval.SELF_RAG_SYSTEM]["verification"]
        self.assertEqual(ver["support_status"], "fully_supported")
        self.assertEqual(ver["usefulness"], "useful")
        self.assertIn("not ground-truth", ver["note"])

    def test_traditional_metrics_unavailable_not_zero(self):
        self.seed()
        detail = live_metrics.build_request_detail(RID, self.db)
        for system in (live_eval.BASELINE_SYSTEM, live_eval.SELF_RAG_SYSTEM):
            block = detail["systems"][system]["traditional_metrics"]
            self.assertEqual(block["status"], "unavailable")
            for name in auto_eval.TRADITIONAL_METRIC_NAMES:
                self.assertIsNone(block[name], name)
        self.assertEqual(detail["traditional_metrics"]["reason"], "requires_relevance_judgments")

    def test_no_manual_labels_are_required(self):
        self.seed()
        self.assertEqual(live_db.count_relevance_labels(path=self.db), 0)
        detail = live_metrics.build_request_detail(RID, self.db)
        self.assertEqual(detail["systems"][live_eval.BASELINE_SYSTEM]["automatic"]["status"], "ok")

    def test_missing_chunk_text_is_insufficient_data_not_zero(self):
        textless = [{"rank": 1, "source": "checkout-api-runbook.md", "text": ""}]
        self.seed(chunks=textless, final=textless)
        detail = live_metrics.build_request_detail(RID, self.db)
        block = detail["systems"][live_eval.SELF_RAG_SYSTEM]["automatic"]
        self.assertEqual(block["status"], "insufficient_data")
        self.assertEqual(block["reason"], "chunk_text_not_recorded")
        self.assertIsNone(block["mean_query_term_coverage"])

    def test_detail_does_not_regenerate_answer_or_duplicate_evaluations(self):
        self.seed()
        with mock.patch("src.self_rag.run_self_rag", side_effect=AssertionError("answer generation must not run")):
            first = live_metrics.build_request_detail(RID, self.db)
            second = live_metrics.build_request_detail(RID, self.db)
        self.assertEqual(
            first["systems"][live_eval.BASELINE_SYSTEM]["answer"],
            second["systems"][live_eval.BASELINE_SYSTEM]["answer"],
        )
        rows = live_db.list_auto_evaluations(request_id=RID, path=self.db)
        self.assertEqual(len(rows), 2)  # one per system, no duplicates
        stored = live_db.get_auto_evaluation(RID, live_eval.BASELINE_SYSTEM, auto_eval.METHOD, self.db)
        self.assertIsNotNone(stored)

    def test_only_one_system_evaluated(self):
        self.seed(baseline=False)
        detail = live_metrics.build_request_detail(RID, self.db)
        self.assertNotIn(live_eval.BASELINE_SYSTEM, detail["systems"])
        self.assertIn(live_eval.SELF_RAG_SYSTEM, detail["systems"])


class HistoryTests(_Base):
    def test_history_reports_per_question_status(self):
        self.seed()
        history = live_metrics.build_history(path=self.db)
        item = history["items"][0]
        self.assertEqual(item["request_id"], RID)
        self.assertEqual(item["evaluation"]["label"], "Both systems evaluated")
        self.assertEqual(item["baseline"]["system_version"], live_eval.BASELINE_SYSTEM)
        self.assertEqual(item["self_rag"]["system_version"], live_eval.SELF_RAG_SYSTEM)

    def test_history_without_evaluations(self):
        live_db.save_request(
            live_eval._request_record(RID, TS, "q", "thread", status="ok", latency_seconds=0.1, result=None),
            self.db,
        )
        item = live_metrics.build_history(path=self.db)["items"][0]
        self.assertEqual(item["evaluation"]["status"], "insufficient_data")
        self.assertIsNone(item["baseline"])


class JudgeTests(_Base):
    def test_judge_not_configured_by_default(self):
        self.seed()
        result = live_metrics.judge_request(RID, live_eval.BASELINE_SYSTEM, self.db)
        self.assertEqual(result["status"], "not_configured")

    def test_judge_bounded_to_one_call_and_stored(self):
        self.seed()
        score = {"method": auto_eval.JUDGE_METHOD, "answer_relevance": 0.8, "faithfulness": 0.7}
        with mock.patch.object(live_metrics.auto_eval, "judge_enabled", return_value=True), \
             mock.patch.object(live_metrics.auto_eval, "judge_answer", return_value=score) as judge:
            first = live_metrics.judge_request(RID, live_eval.BASELINE_SYSTEM, self.db)
            second = live_metrics.judge_request(RID, live_eval.BASELINE_SYSTEM, self.db)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "already_scored")
        self.assertEqual(judge.call_count, 1)

    def test_judge_failure_is_reported_not_fabricated(self):
        self.seed()
        with mock.patch.object(live_metrics.auto_eval, "judge_enabled", return_value=True), \
             mock.patch.object(live_metrics.auto_eval, "judge_answer", side_effect=RuntimeError("boom")):
            result = live_metrics.judge_request(RID, live_eval.BASELINE_SYSTEM, self.db)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("score", result)


class MigrationTests(unittest.TestCase):
    OLD_EVALS_SQL = """
    CREATE TABLE live_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        system_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        error TEXT, error_category TEXT, latency_seconds REAL, answer TEXT, route TEXT,
        retries INTEGER, support_status TEXT, usefulness TEXT,
        used_web_search INTEGER DEFAULT 0, retrieval_available INTEGER DEFAULT 0,
        retrieved_sources_json TEXT, used_sources_json TEXT,
        prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
        token_usage_available INTEGER DEFAULT 0, estimated_cost_usd REAL,
        UNIQUE (request_id, system_version)
    );
    """

    def test_login_columns_are_added_additively(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "audit.db"
            conn = sqlite3.connect(db)
            try:
                conn.executescript(self.OLD_EVALS_SQL)
                conn.execute(
                    "INSERT INTO live_evaluations (request_id, system_version, created_at, status) VALUES (?,?,?,?)",
                    ("r1", "axiom_self_rag", TS, "ok"),
                )
                conn.commit()
            finally:
                conn.close()

            live_db.init_live_db(db)

            conn = sqlite3.connect(db)
            try:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(live_evaluations)")}
                count = conn.execute("SELECT COUNT(*) FROM live_evaluations").fetchone()[0]
            finally:
                conn.close()
            for column in ("retrieved_chunks_json", "final_evidence_json", "grades_json"):
                self.assertIn(column, columns)
            self.assertEqual(count, 1)  # existing row preserved
            stored = live_db.list_evaluations(request_id="r1", path=db)
            self.assertEqual(stored[0]["retrieved_chunks"], [])


class ApiTests(_Base):
    def setUp(self):
        super().setUp()
        self.settings = FakeSettings(self.db)
        self._patcher = mock.patch.object(live_db, "get_settings", lambda: self.settings)
        self._patcher.start()
        self.seed()

    def tearDown(self):
        self._patcher.stop()
        super().tearDown()

    def _client(self):
        app = FastAPI()
        from src.api_eval import router
        app.include_router(router)
        return TestClient(app)

    def test_history_and_detail_endpoints(self):
        with self._client() as client:
            history = client.get("/api/evaluation/history")
            detail = client.get(f"/api/evaluation/requests/{RID}")
            missing = client.get("/api/evaluation/requests/nope")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()["items"][0]["request_id"], RID)
        self.assertEqual(detail.status_code, 200)
        body = detail.json()
        self.assertIn("baseline_rag", body["systems"])
        self.assertIn("axiom_self_rag", body["systems"])
        self.assertEqual(missing.status_code, 404)

    def test_judge_endpoint_503_when_not_configured(self):
        with self._client() as client:
            resp = client.post(
                f"/api/evaluation/requests/{RID}/judge",
                params={"system": live_eval.BASELINE_SYSTEM},
            )
            status = client.get("/api/evaluation/judge/status")
        self.assertEqual(resp.status_code, 503)
        self.assertFalse(status.json()["enabled"])


if __name__ == "__main__":
    unittest.main()
