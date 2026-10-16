"""Offline tests for grader-based ranking metrics.

Two layers are covered:

* ``src.auto_eval.grader_ranking_metrics`` - the exact formulas, cut-off
  behaviour, missing/partial grades and invalid recall denominators.
* ``src.live_metrics`` - Self-RAG grader metrics from its own recorded grades,
  Baseline grade isolation (reuse only on *identical* retrieval), honest
  ``not_evaluated`` states and non-duplicating persistence.

No network/LLM calls: the database is a per-test temporary SQLite file and the
optional grader is exercised only through mocks.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src import auto_eval, live_db, live_eval, live_metrics

TS = "2026-10-01T10:00:00+00:00"
RID = "req-grader"


class FakeSettings:
    """Minimal settings stand-in pointing the database at a temp file."""

    def __init__(self, path):
        self.database_path = str(path)
        self.database_file = Path(path)
        self.live_eval_enabled = True
        self.live_eval_baseline_enabled = True
        self.live_eval_max_concurrency = 1
        self.live_eval_judge_enabled = False
        self.groq_input_cost_per_1m = None
        self.groq_output_cost_per_1m = None


def chunk(rank, source, text, typ="internal"):
    return {
        "rank": rank,
        "source": source,
        "title": source,
        "type": typ,
        "page": None,
        "text": text,
        "chunk_id": auto_eval.chunk_identity(typ, source, text),
    }


def grade(ch, relevant, retrieval_round=1):
    return {
        "chunk_id": ch["chunk_id"],
        "source": ch["source"],
        "title": ch["title"],
        "type": ch["type"],
        "relevant": bool(relevant),
        "rank": ch["rank"],
        "retrieval_round": retrieval_round,
    }


A = chunk(1, "checkout-api-runbook.md", "checkout api returns 502 after deployment")
B = chunk(2, "db-runbook.md", "database connection pool exhausted under load")


# --------------------------------------------------------------------------- #
# Formula correctness
# --------------------------------------------------------------------------- #
class GraderFormulaTests(unittest.TestCase):
    def test_first_rank_relevant_is_perfect(self):
        block = auto_eval.grader_ranking_metrics([A, B], [grade(A, True), grade(B, False)])
        self.assertEqual(block["status"], "ok")
        self.assertEqual(block["precision_at_5"], 0.5)  # 1 of 2 judged considered
        self.assertEqual(block["recall_at_5"], 1.0)  # 1 of 1 judged-relevant
        self.assertEqual(block["reciprocal_rank"], 1.0)
        self.assertEqual(block["map_at_5"], 1.0)
        self.assertEqual(block["ndcg_at_5"], 1.0)
        self.assertTrue(block["complete_judgment"])

    def test_second_rank_relevant_is_discounted(self):
        first = dict(B, rank=1)
        second = dict(A, rank=2)
        block = auto_eval.grader_ranking_metrics(
            [first, second], [grade(first, False), grade(second, True)]
        )
        self.assertEqual(block["precision_at_5"], 0.5)
        self.assertEqual(block["reciprocal_rank"], 0.5)
        self.assertEqual(block["map_at_5"], 0.5)
        # DCG = 1/log2(3); IDCG = 1/log2(2)
        self.assertAlmostEqual(block["ndcg_at_5"], round(1 / 1.5849625007, 4), places=4)

    def test_relevant_at_rank_six_is_outside_top_five(self):
        chunks = [chunk(i, f"runbook-{i}.md", f"filler text number {i}") for i in range(1, 7)]
        chunks[5] = chunk(6, "checkout-api-runbook.md", "checkout api 502 deployment")
        grades = [grade(c, c is chunks[5]) for c in chunks]
        block = auto_eval.grader_ranking_metrics(chunks, grades)
        self.assertEqual(block["considered"], 5)
        self.assertEqual(block["precision_at_5"], 0.0)
        self.assertEqual(block["recall_at_5"], 0.0)
        self.assertEqual(block["reciprocal_rank"], 0.0)
        self.assertEqual(block["map_at_5"], 0.0)
        self.assertEqual(block["ndcg_at_5"], 0.0)

    def test_multiple_relevant_map_and_ndcg(self):
        c1 = chunk(1, "a.md", "alpha one")
        c2 = chunk(2, "b.md", "beta two")
        c3 = chunk(3, "c.md", "gamma three")
        block = auto_eval.grader_ranking_metrics(
            [c2, c1, c3], [grade(c1, True), grade(c2, True), grade(c3, False)]
        )
        # relevant at ranks 1 and 2 -> precisions 1/1 and 2/2 -> AP = 1.0
        self.assertEqual(block["reciprocal_rank"], 1.0)
        self.assertEqual(block["map_at_5"], 1.0)
        dcg = 1 / 1 + 1 / 1.5849625007
        idcg = 1 / 1 + 1 / 1.5849625007
        self.assertAlmostEqual(block["ndcg_at_5"], round(dcg / idcg, 4), places=4)


# --------------------------------------------------------------------------- #
# Missing / partial judgments and invalid denominators
# --------------------------------------------------------------------------- #
class GraderMissingDataTests(unittest.TestCase):
    def test_ungraded_chunk_is_excluded_not_treated_as_irrelevant(self):
        block = auto_eval.grader_ranking_metrics([A, B], [grade(A, True)])
        self.assertEqual(block["graded_total"], 1)
        self.assertEqual(block["ungraded_total"], 1)
        self.assertFalse(block["complete_judgment"])
        # Precision uses only the judged considered chunk -> 1/1, never 1/2.
        self.assertEqual(block["precision_at_5"], 1.0)
        # Recall/MAP/nDCG need a complete scope.
        self.assertIsNone(block["recall_at_5"])
        self.assertIsNone(block["map_at_5"])
        self.assertIsNone(block["ndcg_at_5"])
        self.assertEqual(block["status"], "partial")
        self.assertEqual(block["metric_status"]["precision_at_5"], "ok")
        self.assertEqual(block["metric_status"]["recall_at_5"], "insufficient_data")

    def test_ungraded_before_first_relevant_makes_rr_unavailable(self):
        ungraded = chunk(1, "x.md", "unknown content")
        block = auto_eval.grader_ranking_metrics([ungraded, A], [grade(A, True)])
        self.assertIsNone(block["reciprocal_rank"])  # cannot establish the first relevant

    def test_ungraded_after_first_relevant_keeps_rr(self):
        block = auto_eval.grader_ranking_metrics([A, B], [grade(A, True)])
        self.assertEqual(block["reciprocal_rank"], 1.0)

    def test_no_grades_is_unavailable_not_zero(self):
        block = auto_eval.grader_ranking_metrics([A, B], [])
        self.assertEqual(block["status"], "unavailable")
        self.assertEqual(block["reason"], "no_grader_judgments")
        for name in ("precision_at_5", "recall_at_5", "reciprocal_rank", "map_at_5", "ndcg_at_5"):
            self.assertIsNone(block[name], name)

    def test_no_ranked_retrieval_is_unavailable(self):
        block = auto_eval.grader_ranking_metrics([], [grade(A, True)])
        self.assertEqual(block["status"], "unavailable")
        self.assertEqual(block["reason"], "no_ranked_retrieval")

    def test_zero_relevant_complete_gives_zero_rr_and_no_recall_denominator(self):
        block = auto_eval.grader_ranking_metrics([A, B], [grade(A, False), grade(B, False)])
        self.assertTrue(block["complete_judgment"])
        self.assertEqual(block["precision_at_5"], 0.0)
        self.assertEqual(block["reciprocal_rank"], 0.0)  # complete scope -> genuine zero
        self.assertIsNone(block["recall_at_5"])  # 0 judged-relevant -> invalid denominator
        self.assertIsNone(block["map_at_5"])
        self.assertIsNone(block["ndcg_at_5"])

    def test_grade_matched_by_chunk_id_not_document_name(self):
        # Two chunks share a document name but have different text and grades.
        first = chunk(1, "same.md", "checkout api 502 deployment")
        second = chunk(2, "same.md", "unrelated capacity planning")
        block = auto_eval.grader_ranking_metrics(
            [first, second], [grade(first, False), grade(second, True)]
        )
        self.assertEqual(block["graded_total"], 2)
        self.assertEqual(block["reciprocal_rank"], 0.5)  # the relevant one is at rank 2

    def test_legacy_grade_without_chunk_id_falls_back_to_source(self):
        legacy = {
            "source": A["source"],
            "title": A["title"],
            "type": A["type"],
            "relevant": True,
        }
        block = auto_eval.grader_ranking_metrics([A, B], [legacy])
        self.assertEqual(block["graded_total"], 1)
        self.assertEqual(block["reciprocal_rank"], 1.0)

    def test_same_ranked_retrieval_requires_id_and_text(self):
        self.assertTrue(auto_eval.same_ranked_retrieval([A, B], [dict(A), dict(B)]))
        # Rank is authoritative: the same ranks are identical regardless of list order.
        self.assertTrue(auto_eval.same_ranked_retrieval([A, B], [B, A]))
        # Changed ranks (a different ranked sequence) are not identical.
        reordered = [dict(A, rank=2), dict(B, rank=1)]
        self.assertFalse(auto_eval.same_ranked_retrieval([A, B], reordered))
        altered = dict(A, text="different text")
        self.assertFalse(auto_eval.same_ranked_retrieval([A], [altered]))
        self.assertFalse(auto_eval.same_ranked_retrieval([A], []))


# --------------------------------------------------------------------------- #
# Integration with the persisted live records
# --------------------------------------------------------------------------- #
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

    def seed(self, *, baseline_chunks=None, self_chunks=(A, B), grades=None, baseline=True):
        live_db.save_request(
            live_eval._request_record(
                RID, TS, "how do I fix the checkout api 502 errors after deployment?",
                "thread-1", status="ok", latency_seconds=1.0, result=None,
            ),
            self.db,
        )
        if baseline:
            live_db.save_evaluation(
                live_eval._evaluation_record(
                    RID, live_eval.BASELINE_SYSTEM, TS, status="ok", latency_seconds=1.1,
                    answer="Restart the checkout deployment.",
                    retrieval_available=True,
                    retrieved_chunks=list(baseline_chunks if baseline_chunks is not None else self_chunks),
                ),
                self.db,
            )
        if grades is None:
            grades = [grade(A, True), grade(B, False)]
        live_db.save_evaluation(
            live_eval._evaluation_record(
                RID, live_eval.SELF_RAG_SYSTEM, TS, status="ok", latency_seconds=0.9,
                answer="Check the deployment time.",
                retrieval_available=True,
                retrieved_chunks=list(self_chunks),
                final_evidence=[dict(A)],
                grades=list(grades),
            ),
            self.db,
        )


class SelfRagGraderIntegrationTests(_Base):
    def test_self_rag_grader_metrics_from_recorded_grades(self):
        self.seed()
        detail = live_metrics.build_request_detail(RID, self.db)
        grader = detail["systems"][live_eval.SELF_RAG_SYSTEM]["grader_ranking"]
        self.assertEqual(grader["source"], "self_rag_grader")
        self.assertEqual(grader["precision_at_5"], 0.5)
        self.assertEqual(grader["recall_at_5"], 1.0)
        self.assertEqual(grader["reciprocal_rank"], 1.0)
        self.assertEqual(grader["status"], "ok")

    def test_baseline_reuses_identical_self_rag_grades(self):
        self.seed()
        detail = live_metrics.build_request_detail(RID, self.db)
        baseline = detail["systems"][live_eval.BASELINE_SYSTEM]["grader_ranking"]
        self_rag = detail["systems"][live_eval.SELF_RAG_SYSTEM]["grader_ranking"]
        self.assertEqual(baseline["source"], "baseline_grader_reused")
        self.assertEqual(baseline["precision_at_5"], self_rag["precision_at_5"])
        self.assertEqual(baseline["recall_at_5"], self_rag["recall_at_5"])
        self.assertIn("identical ranked chunks", baseline["reuse_note"])
        # The shared initial retrieval is flagged so the UI can show its text once.
        self.assertTrue(detail["shared_initial_retrieval"])

    def test_baseline_grades_are_not_copied_when_retrieval_differs(self):
        other = chunk(1, "checkout-api-runbook.md", "completely different wording about pods")
        self.seed(baseline_chunks=[other])
        detail = live_metrics.build_request_detail(RID, self.db)
        baseline = detail["systems"][live_eval.BASELINE_SYSTEM]["grader_ranking"]
        self.assertEqual(baseline["status"], "not_evaluated")
        self.assertEqual(baseline["reason"], "baseline_grader_not_run")
        self.assertFalse(detail["shared_initial_retrieval"])
        for name in ("precision_at_5", "recall_at_5", "reciprocal_rank", "map_at_5", "ndcg_at_5"):
            self.assertIsNone(baseline[name], name)

    def test_detail_is_not_duplicated_on_reopen(self):
        self.seed()
        live_metrics.build_request_detail(RID, self.db)
        live_metrics.build_request_detail(RID, self.db)
        rows = [
            r for r in live_db.list_auto_evaluations(request_id=RID, path=self.db)
            if r["method"] == auto_eval.METHOD
        ]
        self.assertEqual(len(rows), 2)  # one per system, never duplicated


class BaselineGraderTests(_Base):
    def test_grader_not_configured_by_default(self):
        self.seed(baseline_chunks=[chunk(1, "checkout-api-runbook.md", "different wording")])
        result = live_metrics.grade_baseline_request(RID, self.db)
        self.assertEqual(result["status"], "not_configured")

    def test_grader_stores_once_and_recomputes_baseline(self):
        other = chunk(1, "checkout-api-runbook.md", "different wording about pods")
        self.seed(baseline_chunks=[other])
        ranked = [grade(other, True)]
        with mock.patch.object(live_metrics.auto_eval, "judge_enabled", return_value=True), \
             mock.patch.object(live_metrics.auto_eval, "grade_chunks", return_value=ranked) as grader:
            first = live_metrics.grade_baseline_request(RID, self.db)
            second = live_metrics.grade_baseline_request(RID, self.db)
            detail = live_metrics.build_request_detail(RID, self.db)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "already_graded")
        self.assertEqual(grader.call_count, 1)
        baseline = detail["systems"][live_eval.BASELINE_SYSTEM]["grader_ranking"]
        self.assertEqual(baseline["source"], "baseline_grader_llm")
        self.assertEqual(baseline["precision_at_5"], 1.0)

    def test_grader_failure_is_reported_not_fabricated(self):
        other = chunk(1, "checkout-api-runbook.md", "different wording about pods")
        self.seed(baseline_chunks=[other])
        with mock.patch.object(live_metrics.auto_eval, "judge_enabled", return_value=True), \
             mock.patch.object(live_metrics.auto_eval, "grade_chunks", side_effect=RuntimeError("boom")):
            result = live_metrics.grade_baseline_request(RID, self.db)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("grades", result)


class GraderApiTests(_Base):
    def setUp(self):
        super().setUp()
        self.settings = FakeSettings(self.db)
        self._patcher = mock.patch.object(live_db, "get_settings", lambda: self.settings)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        super().tearDown()

    def _client(self):
        app = FastAPI()
        from src.api_eval import router
        app.include_router(router)
        return TestClient(app)

    def test_grade_baseline_endpoint_503_when_not_configured(self):
        self.seed(baseline_chunks=[chunk(1, "checkout-api-runbook.md", "different wording")])
        with self._client() as client:
            resp = client.post(f"/api/evaluation/requests/{RID}/grade-baseline")
            missing = client.post("/api/evaluation/requests/nope/grade-baseline")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(missing.status_code, 404)


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):  # noqa: ARG002 - mirrors the retriever interface
        return list(self._docs)


class _FakeBaseline:
    def run(self, question, docs):  # noqa: ARG002
        return {
            "answer": "Restart the checkout deployment.",
            "route": "internal_retrieval",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class EndToEndGraderTests(_Base):
    """The full record_chat -> baseline-shadow -> detail path, no network."""

    def test_baseline_reuses_self_rag_grades_through_real_paths(self):
        docs = [
            Document(page_content=A["text"], metadata={"document_name": A["source"]}),
            Document(page_content=B["text"], metadata={"document_name": B["source"]}),
        ]
        result = {
            "answer": "Check the deployment time.",
            "route": "Private Runbooks",
            "used_web_search": False,
            "support_status": "fully_supported",
            "usefulness": "useful",
            "sources": [{"type": "internal", "title": A["source"], "source": A["source"]}],
            "retrieved_chunks": [dict(A), dict(B)],
            "final_evidence": [dict(A)],
            "grades": [grade(A, True), grade(B, False)],
            "retrieval_available": True,
            "trace": ["Internal retrieval: 2 chunks", "Relevance grade (internal): 1/2 relevant"],
        }
        live_eval.record_chat(
            RID,
            "how do I fix the checkout api 502 errors after deployment?",
            "thread-1",
            result,
            1.0,
            path=self.db,
        )
        with mock.patch.object(live_eval, "_get_retriever", lambda: _FakeRetriever(docs)), \
             mock.patch.object(live_eval, "_get_baseline", lambda: _FakeBaseline()):
            live_eval.run_baseline_shadow(
                RID, "how do I fix the checkout api 502 errors after deployment?", path=self.db
            )

        detail = live_metrics.build_request_detail(RID, self.db)
        self_rag = detail["systems"][live_eval.SELF_RAG_SYSTEM]["grader_ranking"]
        baseline = detail["systems"][live_eval.BASELINE_SYSTEM]["grader_ranking"]
        self.assertEqual(self_rag["source"], "self_rag_grader")
        self.assertEqual(self_rag["precision_at_5"], 0.5)
        # Baseline retrieved the identical chunks, so grades are reused (verified).
        self.assertEqual(baseline["source"], "baseline_grader_reused")
        self.assertEqual(baseline["precision_at_5"], 0.5)
        # Baseline evidence is not duplicated: it is its retrieval, exposed once.
        self.assertEqual(detail["systems"][live_eval.BASELINE_SYSTEM]["final_evidence"], [])


if __name__ == "__main__":
    unittest.main()
