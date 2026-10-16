"""Offline unit tests for the deterministic automatic evaluator.

No network, no model: ``src.auto_eval`` is pure and local. The checks below cover
the lexical proxies, the explicit unavailable/insufficient states (never zero),
and that the optional judge stays off unless configured.
"""

import unittest

from src import auto_eval


QUESTION = "checkout api 502 errors after deployment"
CHUNK_1 = {
    "rank": 1,
    "source": "checkout-api-runbook.md",
    "title": "Checkout API Runbook",
    "text": "checkout api returns 502 after deployment",
}
CHUNK_2 = {
    "rank": 2,
    "source": "db-runbook.md",
    "title": "DB Runbook",
    "text": "database connection pool exhausted",
}


class TokenizationTests(unittest.TestCase):
    def test_stopwords_and_short_tokens_removed(self):
        terms = auto_eval.content_terms("The 502 error is on the API a b")
        self.assertIn("502", terms)
        self.assertIn("error", terms)
        self.assertIn("api", terms)
        self.assertNotIn("the", terms)
        self.assertNotIn("is", terms)
        self.assertNotIn("a", terms)


class RetrievalProxyTests(unittest.TestCase):
    def test_query_term_coverage_over_chunks(self):
        block = auto_eval.evaluate_retrieval(QUESTION, [CHUNK_1, CHUNK_2])
        self.assertEqual(block["status"], "ok")
        self.assertEqual(block["chunks_retrieved"], 2)
        self.assertEqual(block["chunks_with_text"], 2)
        self.assertEqual(block["distinct_sources"], 2)
        self.assertEqual(block["top_rank_query_term_coverage"], 0.8)
        self.assertEqual(block["mean_query_term_coverage"], 0.4)
        self.assertEqual(block["query_terms_covered_by_any_chunk"], 0.8)
        self.assertIsNone(block["mean_retrieval_score"])  # retriever exposed none

    def test_mean_retrieval_score_used_when_present(self):
        chunks = [{**CHUNK_1, "score": 0.5}, {**CHUNK_2, "score": 0.25}]
        block = auto_eval.evaluate_retrieval(QUESTION, chunks)
        self.assertEqual(block["mean_retrieval_score"], 0.375)

    def test_no_chunks_is_unavailable_not_zero(self):
        block = auto_eval.evaluate_retrieval(QUESTION, [])
        self.assertEqual(block["status"], "unavailable")
        self.assertEqual(block["reason"], "no_retrieved_chunks")
        self.assertIsNone(block["mean_query_term_coverage"])
        self.assertIsNone(block["query_terms_covered_by_any_chunk"])

    def test_chunks_without_text_is_insufficient_data(self):
        block = auto_eval.evaluate_retrieval(QUESTION, [{"rank": 1, "source": "a.md"}])
        self.assertEqual(block["status"], "insufficient_data")
        self.assertEqual(block["reason"], "chunk_text_not_recorded")
        self.assertIsNone(block["mean_query_term_coverage"])
        self.assertEqual(block["chunks_retrieved"], 1)

    def test_normalize_defaults_rank_and_keeps_missing_score_null(self):
        normalized = auto_eval.normalize_chunks([{"source": "b.md"}, {"source": "a.md", "title": "A"}])
        self.assertEqual([c["rank"] for c in normalized], [1, 2])
        self.assertTrue(all(c["score"] is None for c in normalized))


class AnswerProxyTests(unittest.TestCase):
    def test_answer_relevance_and_evidence_support(self):
        block = auto_eval.evaluate_answer(QUESTION, "Restart the checkout api deployment", [CHUNK_1])
        self.assertEqual(block["status"], "ok")
        self.assertEqual(block["answer_relevance"], 0.6)
        self.assertEqual(block["evidence_support"], 0.75)
        self.assertEqual(block["claims_total"], 1)
        self.assertEqual(block["unsupported_claims"], 0)
        self.assertEqual(block["unsupported_terms"], ["restart"])

    def test_unsupported_claim_detected(self):
        answer = "Restart the checkout api deployment. The moon is made of cheese."
        block = auto_eval.evaluate_answer(QUESTION, answer, [CHUNK_1])
        self.assertEqual(block["claims_total"], 2)
        self.assertEqual(block["unsupported_claims"], 1)

    def test_no_answer_is_unavailable(self):
        block = auto_eval.evaluate_answer(QUESTION, "", [CHUNK_1])
        self.assertEqual(block["status"], "unavailable")
        self.assertEqual(block["reason"], "no_answer")
        self.assertIsNone(block["answer_relevance"])

    def test_no_evidence_is_insufficient_data_not_zero(self):
        block = auto_eval.evaluate_answer(QUESTION, "Restart the checkout api deployment", [])
        self.assertEqual(block["status"], "insufficient_data")
        self.assertEqual(block["reason"], "no_evidence_text_recorded")
        self.assertIsNone(block["evidence_support"])
        self.assertIsNone(block["unsupported_claims"])


class TraditionalMetricTests(unittest.TestCase):
    def test_traditional_metrics_are_unavailable_not_zero(self):
        block = auto_eval.traditional_metrics_unavailable()
        self.assertEqual(block["status"], "unavailable")
        self.assertEqual(block["reason"], "requires_relevance_judgments")
        for name in auto_eval.TRADITIONAL_METRIC_NAMES:
            self.assertIsNone(block[name], name)

    def test_evaluate_system_shape(self):
        result = auto_eval.evaluate_system(
            QUESTION, "Restart the checkout api deployment", [CHUNK_1], [CHUNK_1],
            verification={"support_status": "fully_supported", "usefulness": "useful", "retries": 0},
        )
        self.assertEqual(result["method"], auto_eval.METHOD)
        self.assertEqual(result["retrieval"]["status"], "ok")
        self.assertEqual(result["answer"]["status"], "ok")
        self.assertEqual(result["verification"]["support_status"], "fully_supported")
        self.assertEqual(result["traditional_metrics"]["status"], "unavailable")


class JudgeGateTests(unittest.TestCase):
    class _Settings:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def test_judge_disabled_by_default(self):
        self.assertFalse(auto_eval.judge_enabled(self._Settings()))

    def test_judge_requires_flag_and_key(self):
        self.assertFalse(auto_eval.judge_enabled(self._Settings(live_eval_judge_enabled=True, groq_api_key="")))
        self.assertTrue(auto_eval.judge_enabled(self._Settings(live_eval_judge_enabled=True, groq_api_key="k")))

    def test_judge_answer_without_key_raises(self):
        with self.assertRaises(RuntimeError):
            auto_eval.judge_answer("q", "a", ["e"], settings=self._Settings(groq_api_key=""))


if __name__ == "__main__":
    unittest.main()
