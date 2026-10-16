"""Offline tests for live request logging, shadow baseline evaluation, metrics
and the dashboard API.

No network calls: the Self-RAG graph, retriever and Baseline RAG are all mocked,
and the database is a per-test temporary SQLite file.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app as app_module
from src import live_db, live_eval, live_metrics


class FakeSettings:
    """Minimal settings stand-in pointing the DB at a temp file."""

    def __init__(self, path, *, live_eval_enabled=True, baseline_enabled=False):
        self.database_path = str(path)
        self.database_file = Path(path)
        self.live_eval_enabled = live_eval_enabled
        self.live_eval_baseline_enabled = baseline_enabled
        self.live_eval_max_concurrency = 1
        self.groq_input_cost_per_1m = None
        self.groq_output_cost_per_1m = None


def _result(**kwargs):
    base = {
        "answer": "Compare the incident start with the deployment time.",
        "route": "Private Runbooks",
        "used_web_search": False,
        "support_status": "fully_supported",
        "usefulness": "useful",
        "sources": [
            {"type": "internal", "title": "checkout-api-runbook.md", "source": "checkout-api-runbook.md", "url": None, "page": 2}
        ],
        "retrieved_chunks": [
            {"rank": 1, "source": "checkout-api-runbook.md", "title": "Checkout API Runbook",
             "type": "internal", "page": 2, "text": "restart the pod after checking readiness"},
        ],
        "final_evidence": [
            {"rank": 1, "source": "checkout-api-runbook.md", "title": "Checkout API Runbook",
             "type": "internal", "page": 2, "text": "restart the pod after checking readiness"},
        ],
        "grades": [
            {"source": "checkout-api-runbook.md", "title": "Checkout API Runbook", "type": "internal", "relevant": True},
        ],
        "retrieval_available": True,
        "trace": [
            "Retrieval decision: True",
            "Internal retrieval: 3 chunks",
            "Relevance grade (internal): 2/3 relevant",
            "Generated answer from internal evidence",
            "Support check: fully_supported",
            "Usefulness check: useful",
        ],
        "thread_id": "t-1",
        "memory_turns": 1,
    }
    base.update(kwargs)
    return base


class _FakeRetriever:
    def __init__(self, docs=None, raise_exc=None):
        self._docs = docs if docs is not None else [
            Document(page_content="restart the pod after checking readiness",
                     metadata={"document_name": "checkout-api-runbook.md"})
        ]
        self._raise = raise_exc

    def invoke(self, query):
        if self._raise:
            raise self._raise
        return self._docs


class _FakeBaseline:
    def __init__(self, *, raise_exc=None):
        self._raise = raise_exc

    def run(self, question, docs):
        if self._raise:
            raise self._raise
        return {
            "answer": "Restart the checkout-api deployment.",
            "route": "internal_retrieval",
            "retries": 0,
            "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
        }


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

    # -- record helpers built through the real code paths ------------------
    def add_request(self, rid, created_at, question="What should I check?", status="ok", latency=1.0):
        record = live_eval._request_record(
            rid,
            created_at,
            question,
            "thread-1",
            status=status,
            latency_seconds=latency,
            result=_result() if status == "ok" else None,
            error=None if status == "ok" else "provider down",
            error_category=None if status == "ok" else "application",
        )
        live_db.save_request(record, self.db)
        return record

    def add_eval(self, rid, system, created_at, status="ok", latency=1.0, **kwargs):
        record = live_eval._evaluation_record(
            rid, system, created_at, status=status, latency_seconds=latency, **kwargs
        )
        live_db.save_evaluation(record, self.db)
        return record


# --------------------------------------------------------------------------- #
# 1. Request logging and persistence
# --------------------------------------------------------------------------- #
class RequestLoggingTests(_Base):
    def test_record_chat_persists_request_and_self_rag_evaluation(self):
        live_eval.record_chat("rid-1", "what should I check?", "thread-1", _result(), 1.234, path=self.db)

        request = live_db.get_request("rid-1", self.db)
        self.assertIsNotNone(request)
        self.assertEqual(request["status"], "ok")
        self.assertEqual(request["latency_seconds"], 1.234)
        self.assertEqual(request["support_status"], "fully_supported")
        # Retry marker count comes from the trace ("Rewrote..." markers absent -> 0).
        self.assertEqual(request["retries"], 0)
        self.assertEqual(request["sources"][0]["title"], "checkout-api-runbook.md")

        evals = live_db.list_evaluations(request_id="rid-1", path=self.db)
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["system_version"], live_eval.SELF_RAG_SYSTEM)

    def test_record_chat_failure_is_persisted_with_category(self):
        live_eval.record_chat(
            "rid-2", "q", "thread-1", None, 0.5, ValueError("429 rate limit quota"), path=self.db
        )
        request = live_db.get_request("rid-2", self.db)
        self.assertEqual(request["status"], "failed")
        self.assertEqual(request["error_category"], "quota_or_rate_limit")
        evals = live_db.list_evaluations(request_id="rid-2", path=self.db)
        self.assertEqual(evals[0]["status"], "failed")

    def test_record_chat_never_raises_when_storage_fails(self):
        with mock.patch.object(live_db, "save_request", side_effect=RuntimeError("disk full")):
            # Must not raise.
            live_eval.record_chat("rid-3", "q", "thread-1", _result(), 1.0, path=self.db)

    def test_record_chat_disabled_writes_nothing(self):
        settings = FakeSettings(self.db, live_eval_enabled=False)
        with mock.patch.object(live_eval, "get_settings", lambda: settings):
            live_eval.record_chat("rid-4", "q", "thread-1", _result(), 1.0, path=self.db)
        self.assertIsNone(live_db.get_request("rid-4", self.db))


# --------------------------------------------------------------------------- #
# 2. Baseline shadow evaluation
# --------------------------------------------------------------------------- #
class BaselineShadowTests(_Base):
    def _run(self, rid, question="q"):
        with mock.patch.object(live_eval, "_get_retriever", lambda: _FakeRetriever()), \
             mock.patch.object(live_eval, "_get_baseline", lambda: _FakeBaseline()):
            return live_eval.run_baseline_shadow(rid, question, path=self.db)

    def test_both_systems_share_request_id_and_baseline_is_idempotent(self):
        live_eval.record_chat("rid-10", "q", "thread-1", _result(), 2.0, path=self.db)
        record = self._run("rid-10")
        self.assertEqual(record["system_version"], live_eval.BASELINE_SYSTEM)
        self.assertEqual(record["request_id"], "rid-10")

        evals = live_db.list_evaluations(request_id="rid-10", path=self.db)
        systems = sorted(e["system_version"] for e in evals)
        self.assertEqual(systems, ["axiom_self_rag", "baseline_rag"])

        # Second run is a no-op (no duplicate execution).
        self.assertIsNone(self._run("rid-10"))
        evals = live_db.list_evaluations(request_id="rid-10", path=self.db)
        self.assertEqual(len(evals), 2)

    def test_shared_raw_retrieval_recorded_on_request(self):
        self.add_request("rid-11", "2026-10-01T10:00:00+00:00")
        self._run("rid-11")
        request = live_db.get_request("rid-11", self.db)
        self.assertTrue(request["retrieval_available"])
        self.assertEqual(request["retrieved_sources"], ["checkout-api-runbook.md"])

    def test_baseline_failure_recorded_not_raised(self):
        self.add_request("rid-12", "2026-10-01T10:00:00+00:00")
        with mock.patch.object(live_eval, "_get_retriever", lambda: _FakeRetriever(raise_exc=RuntimeError("connection reset by peer"))):
            record = live_eval.run_baseline_shadow("rid-12", "q", path=self.db)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_category"], "network")
        stored = live_db.list_evaluations(request_id="rid-12", path=self.db)
        self.assertEqual(stored[0]["status"], "failed")

    def test_disabled_baseline_skips(self):
        settings = FakeSettings(self.db, baseline_enabled=False)
        with mock.patch.object(live_eval, "get_settings", lambda: settings):
            self.assertIsNone(live_eval.run_baseline_shadow("rid-13", "q", path=self.db))
        self.assertEqual(live_db.list_evaluations(request_id="rid-13", path=self.db), [])


# --------------------------------------------------------------------------- #
# 3. Metrics calculations and missing-data states
# --------------------------------------------------------------------------- #
class MetricsTests(_Base):
    def test_empty_database_reports_insufficient_data_not_zero(self):
        summary = live_metrics.build_summary(path=self.db)
        self.assertEqual(summary["requests"]["total"], 0)
        self.assertIsNone(summary["requests"]["success_rate"])
        self.assertEqual(summary["requests"]["status"], "insufficient_data")
        self.assertIsNone(summary["latency"]["average_seconds"])
        self.assertEqual(summary["latency"]["status"], "insufficient_data")
        self.assertEqual(summary["quality"]["answer"]["status"], "insufficient_data")
        self.assertEqual(summary["self_rag"]["status"], "insufficient_data")

    def test_summary_counts_latency_and_quality_pending(self):
        self.add_request("r1", "2026-10-01T10:00:00+00:00", latency=2.0)
        self.add_request("r2", "2026-10-01T11:00:00+00:00", latency=4.0)
        self.add_request("r3", "2026-10-02T09:00:00+00:00", status="failed", latency=0.2)
        self.add_eval("r1", live_eval.SELF_RAG_SYSTEM, "2026-10-01T10:00:00+00:00", support_status="fully_supported", usefulness="useful", retries=0)
        self.add_eval("r2", live_eval.SELF_RAG_SYSTEM, "2026-10-01T11:00:00+00:00", support_status="partially_supported", usefulness="useful", retries=1)
        self.add_eval("r3", live_eval.SELF_RAG_SYSTEM, "2026-10-02T09:00:00+00:00", status="failed")

        summary = live_metrics.build_summary(path=self.db)
        self.assertEqual(summary["requests"]["total"], 3)
        self.assertEqual(summary["requests"]["evaluated"], 3)
        self.assertEqual(summary["requests"]["success"], 2)
        self.assertEqual(summary["requests"]["failed"], 1)
        self.assertAlmostEqual(summary["requests"]["success_rate"], 0.6667, places=3)
        # Latency uses successful requests only -> 2.0 and 4.0.
        self.assertAlmostEqual(summary["latency"]["average_seconds"], 3.0, places=4)
        self.assertAlmostEqual(summary["latency"]["p95_seconds"], 3.9, places=3)
        self.assertEqual(summary["latency"]["sample"], 2)
        # No quality scores -> pending (requests exist), never 0.
        self.assertEqual(summary["quality"]["answer"]["status"], "pending")
        self.assertIsNone(summary["quality"]["answer"]["answer_correctness"]["mean"])
        # Tokens unavailable (Self-RAG does not expose usage here).
        self.assertFalse(summary["tokens"]["available"])
        self.assertEqual(summary["tokens"]["status"], "unavailable")

    def test_self_rag_workflow_rates(self):
        for i in range(4):
            self.add_request(f"w{i}", f"2026-10-0{i+1}T10:00:00+00:00")
        self.add_eval("w0", live_eval.SELF_RAG_SYSTEM, "2026-10-01T10:00:00+00:00", support_status="fully_supported", usefulness="useful", retries=0, used_web_search=False)
        self.add_eval("w1", live_eval.SELF_RAG_SYSTEM, "2026-10-02T10:00:00+00:00", support_status="partially_supported", usefulness="useful", retries=2, used_web_search=True)
        self.add_eval("w2", live_eval.SELF_RAG_SYSTEM, "2026-10-03T10:00:00+00:00", support_status="no_support", usefulness="not_useful", retries=0, used_web_search=False)
        self.add_eval("w3", live_eval.SELF_RAG_SYSTEM, "2026-10-04T10:00:00+00:00", support_status="fully_supported", usefulness="useful", retries=0, used_web_search=False)

        wf = live_metrics.build_summary(path=self.db)["self_rag"]
        self.assertAlmostEqual(wf["retry_rate"], 0.25, places=4)
        self.assertAlmostEqual(wf["web_search_rate"], 0.25, places=4)
        self.assertAlmostEqual(wf["verification_rate"], 1.0, places=4)
        self.assertEqual(wf["support_status"]["fully_supported"], 2)
        self.assertEqual(wf["usefulness"]["not_useful"], 1)

    def test_comparison_scores_both_systems_with_same_rules(self):
        self.add_request("c1", "2026-10-01T10:00:00+00:00")
        self.add_eval("c1", live_eval.SELF_RAG_SYSTEM, "2026-10-01T10:00:00+00:00", status="ok")
        self.add_eval("c1", live_eval.BASELINE_SYSTEM, "2026-10-01T10:00:00+00:00", status="failed",
                      error="boom", error_category="application",
                      prompt_tokens=100, completion_tokens=10, total_tokens=110, token_usage_available=True,
                      estimated_cost_usd=0.0002)

        comparison = live_metrics.build_comparison(path=self.db)
        self.assertEqual(comparison["systems"][live_eval.SELF_RAG_SYSTEM]["success_rate"], 1.0)
        self.assertEqual(comparison["systems"][live_eval.BASELINE_SYSTEM]["success_rate"], 0.0)
        self.assertTrue(comparison["systems"][live_eval.BASELINE_SYSTEM]["tokens"]["available"])
        self.assertEqual(comparison["systems"][live_eval.BASELINE_SYSTEM]["tokens"]["total_tokens"], 110)
        self.assertAlmostEqual(
            comparison["systems"][live_eval.BASELINE_SYSTEM]["tokens"]["estimated_cost_usd"], 0.0002
        )
        # A system with no rows is explicitly insufficient_data, not zero.
        self.assertEqual(
            live_metrics._system_block([], live_eval.BASELINE_SYSTEM, self.db)["status"], "insufficient_data"
        )

    def test_retrieval_quality_is_automatic_and_has_no_label_dependency(self):
        self.add_request("q1", "2026-10-01T10:00:00+00:00")
        live_db.set_request_retrieval("q1", ["checkout-api-runbook.md"], True, self.db)
        self.add_eval("q1", live_eval.SELF_RAG_SYSTEM, "2026-10-01T10:00:00+00:00")
        # No manual labels exist; the summary reports an automatic aggregate (here
        # empty because no automatic evaluation was stored) and traditional
        # retrieval metrics are explicitly unavailable - never a fabricated score.
        quality = live_metrics.build_summary(path=self.db)["quality"]["retrieval"]
        self.assertEqual(quality["status"], "insufficient_data")
        self.assertEqual(live_db.count_relevance_labels(path=self.db), 0)
        self.assertEqual(quality["traditional_metrics"]["status"], "unavailable")
        self.assertIsNone(quality["traditional_metrics"]["precision_at_5"])

    def test_estimated_cost_only_when_priced(self):
        self.assertEqual(live_eval.estimate_cost(1000, 1000), None)  # no pricing configured
        settings = FakeSettings(self.db)
        settings.groq_input_cost_per_1m = 0.15
        settings.groq_output_cost_per_1m = 0.60
        with mock.patch.object(live_eval, "get_settings", lambda: settings):
            cost = live_eval.estimate_cost(1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.75, places=6)

    def test_daily_groups_by_date(self):
        self.add_request("d1", "2026-10-01T10:00:00+00:00", latency=1.0)
        self.add_request("d2", "2026-10-01T12:00:00+00:00", latency=3.0)
        self.add_request("d3", "2026-10-02T10:00:00+00:00", status="failed", latency=0.1)
        daily = live_metrics.build_daily(path=self.db)
        self.assertEqual([p["date"] for p in daily["points"]], ["2026-10-01", "2026-10-02"])
        self.assertEqual(daily["points"][0]["requests"], 2)
        self.assertAlmostEqual(daily["points"][0]["latency"]["average_seconds"], 2.0, places=4)
        self.assertEqual(daily["points"][1]["failed"], 1)

    def test_history_pagination_and_status_filter(self):
        for i in range(5):
            self.add_request(
                f"h{i}",
                f"2026-10-0{i+1}T10:00:00+00:00",
                status="failed" if i == 4 else "ok",
            )
        page1 = live_metrics.build_history(page=1, page_size=2, path=self.db)
        self.assertEqual(page1["total"], 5)
        self.assertEqual(page1["total_pages"], 3)
        self.assertEqual(len(page1["items"]), 2)
        self.assertTrue(page1["has_next"])
        self.assertFalse(page1["has_prev"])

        page3 = live_metrics.build_history(page=3, page_size=2, path=self.db)
        self.assertEqual(len(page3["items"]), 1)
        self.assertFalse(page3["has_next"])

        failed = live_metrics.build_history(status="failed", path=self.db)
        self.assertEqual(failed["total"], 1)
        self.assertEqual(failed["items"][0]["request_id"], "h4")

    def test_request_detail_reports_failure_and_quality(self):
        self.add_request("x1", "2026-10-01T10:00:00+00:00", status="failed")
        self.add_eval("x1", live_eval.SELF_RAG_SYSTEM, "2026-10-01T10:00:00+00:00",
                      status="failed", error="parser blew up", error_category="parser")
        detail = live_metrics.build_request_detail("x1", self.db)
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(detail["error_category"], "application")
        self.assertIn(live_eval.SELF_RAG_SYSTEM, detail["systems"])
        self.assertEqual(detail["systems"][live_eval.SELF_RAG_SYSTEM]["error_category"], "parser")
        self.assertEqual(detail["quality_status"], "pending")
        self.assertIsNone(live_metrics.build_request_detail("missing", self.db))


# --------------------------------------------------------------------------- #
# 4. Dashboard API (filtering, pagination, error isolation)
# --------------------------------------------------------------------------- #
class ApiTests(_Base):
    def setUp(self):
        super().setUp()
        self.settings = FakeSettings(self.db)
        self.patchers = [
            mock.patch.object(live_db, "get_settings", lambda: self.settings),
            mock.patch.object(live_eval, "get_settings", lambda: self.settings),
            mock.patch.object(app_module, "init_db", lambda: None),
            mock.patch.object(app_module, "save_audit", lambda *a, **k: None),
            mock.patch.object(app_module, "run_self_rag", lambda q, t: _result()),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        super().tearDown()

    def test_chat_logs_request_and_returns_request_id(self):
        with TestClient(app_module.app) as client:
            response = client.post("/api/chat", json={"question": "what should I check?", "thread_id": "thread-1"})
        self.assertEqual(response.status_code, 200)
        request_id = response.json()["request_id"]
        self.assertTrue(request_id)
        stored = live_db.get_request(request_id, self.db)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["question"], "what should I check?")

    def test_logging_failure_does_not_break_chat(self):
        with mock.patch.object(app_module, "record_chat", side_effect=RuntimeError("db unavailable")):
            with TestClient(app_module.app) as client:
                response = client.post("/api/chat", json={"question": "still answer me", "thread_id": "thread-2"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["answer"])

    def test_chat_failure_is_logged_and_returns_500(self):
        def boom(q, t):
            raise RuntimeError("model exploded")

        with mock.patch.object(app_module, "run_self_rag", boom):
            with TestClient(app_module.app) as client:
                response = client.post("/api/chat", json={"question": "will fail", "thread_id": "thread-3"})
        self.assertEqual(response.status_code, 500)
        requests = live_db.list_requests(path=self.db)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["status"], "failed")

    def test_summary_endpoint_date_filter_and_invalid_date(self):
        self.add_request("a1", "2026-10-01T10:00:00+00:00", latency=1.0)
        self.add_request("a2", "2026-10-05T10:00:00+00:00", latency=3.0)
        with TestClient(app_module.app) as client:
            all_resp = client.get("/api/evaluation/summary")
            filtered = client.get("/api/evaluation/summary", params={"date_from": "2026-10-04"})
            bad = client.get("/api/evaluation/summary", params={"date_from": "nope"})
        self.assertEqual(all_resp.status_code, 200)
        self.assertEqual(all_resp.json()["requests"]["total"], 2)
        self.assertEqual(filtered.json()["requests"]["total"], 1)
        self.assertEqual(bad.status_code, 400)

    def test_history_endpoint_pagination(self):
        for i in range(3):
            self.add_request(f"p{i}", f"2026-10-0{i+1}T10:00:00+00:00")
        with TestClient(app_module.app) as client:
            page1 = client.get("/api/evaluation/history", params={"page": 1, "page_size": 2})
            page2 = client.get("/api/evaluation/history", params={"page": 2, "page_size": 2})
            bad = client.get("/api/evaluation/history", params={"page_size": 500})
        self.assertEqual(page1.status_code, 200)
        self.assertEqual(page1.json()["total"], 3)
        self.assertEqual(len(page1.json()["items"]), 2)
        self.assertEqual(len(page2.json()["items"]), 1)
        self.assertEqual(bad.status_code, 422)

    def test_request_detail_endpoint_and_404(self):
        self.add_request("detail-1", "2026-10-01T10:00:00+00:00")
        with TestClient(app_module.app) as client:
            ok = client.get("/api/evaluation/requests/detail-1")
            missing = client.get("/api/evaluation/requests/does-not-exist")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["request_id"], "detail-1")
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
