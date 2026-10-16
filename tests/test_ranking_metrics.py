"""Offline unit tests for question-level ranking metrics (MRR / MAP@5 / nDCG@5).

Standard-library ``unittest`` only and no network calls. These formulas are the
shared offline-benchmark implementations in ``evaluation.metrics`` (the live
dashboard does not compute ground-truth ranking metrics - it has no relevance
reference and reports them as unavailable).
"""

import unittest

from evaluation import metrics


def _row(**kwargs):
    base = {
        "system_version": "baseline_rag",
        "question_id": "Q",
        "category": "direct",
        "retrieved_sources": "",
        "used_sources": "",
        "retrieval_available": True,
        "ground_truth_sources": "",
        "latency_seconds": "1.0",
        "request_failed": "False",
        "error_category": "",
        "token_usage_available": "False",
        "prompt_tokens": "",
        "completion_tokens": "",
        "total_tokens": "",
        "answer_correctness_manual": "",
        "faithfulness_manual": "",
    }
    base.update(kwargs)
    return base


# --------------------------------------------------------------------------- #
# Formula correctness
# --------------------------------------------------------------------------- #
class RankingFormulaTests(unittest.TestCase):
    def test_rank_one_relevant_is_perfect(self):
        m = metrics.ranking_metrics(["a.md", "b.md"], ["a.md"])
        self.assertEqual(m["status"], "scored")
        self.assertEqual(m["precision"], 0.5)  # 1 of 2 distinct retrieved
        self.assertEqual(m["recall"], 1.0)  # 1 of 1 ground truth
        self.assertEqual(m["reciprocal_rank"], 1.0)
        self.assertEqual(m["average_precision"], 1.0)
        self.assertEqual(m["ndcg"], 1.0)

    def test_rank_two_relevant_discounts(self):
        m = metrics.ranking_metrics(["b.md", "a.md"], ["a.md"])
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["reciprocal_rank"], 0.5)
        self.assertEqual(m["average_precision"], 0.5)
        # DCG = 1/log2(3); IDCG = 1/log2(2)
        self.assertAlmostEqual(m["ndcg"], round(1 / 1.5849625007, 4), places=4)

    def test_map_and_ndcg_multiple_relevant(self):
        m = metrics.ranking_metrics(["b.md", "x.md", "a.md"], ["a.md", "b.md"])
        self.assertEqual(m["reciprocal_rank"], 1.0)
        # precisions at ranks 1 and 3: 1/1 and 2/3 -> mean / min(2,5)
        self.assertAlmostEqual(m["average_precision"], round((1 + 2 / 3) / 2, 4), places=4)
        self.assertEqual(m["recall"], 1.0)
        # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1/log2(2) + 1/log2(3)
        self.assertAlmostEqual(m["ndcg"], round(1.5 / (1 + 1 / 1.5849625007), 4), places=4)

    def test_order_changes_reciprocal_rank_and_ap(self):
        first = metrics.ranking_metrics(["a.md", "b.md"], ["a.md"])
        later = metrics.ranking_metrics(["b.md", "a.md"], ["a.md"])
        self.assertGreater(first["reciprocal_rank"], later["reciprocal_rank"])
        self.assertGreater(first["average_precision"], later["average_precision"])
        self.assertGreater(first["ndcg"], later["ndcg"])

    def test_top_k_is_capped_at_five(self):
        # The only relevant document sits at position 6 -> outside the top-5.
        m = metrics.ranking_metrics(["b", "c", "d", "e", "f", "a.md"], ["a.md"], k=5)
        self.assertEqual(m["retrieved_raw_count"], 5)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["reciprocal_rank"], 0.0)


# --------------------------------------------------------------------------- #
# Deduplication, fewer-than-five, empty, missing
# --------------------------------------------------------------------------- #
class RankingEdgeCaseTests(unittest.TestCase):
    def test_duplicate_ids_are_deduplicated_deterministically(self):
        m = metrics.ranking_metrics(["a.md", "a.md", "a.md", "b.md"], ["a.md"])
        self.assertEqual(m["retrieved_count"], 2)  # distinct
        self.assertEqual(m["retrieved_raw_count"], 4)  # raw (capped at k)
        self.assertEqual(m["denominator"], 2)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["reciprocal_rank"], 1.0)  # first occurrence wins

    def test_fewer_than_five_retrieved_is_not_padded(self):
        m = metrics.ranking_metrics(["b", "c"], ["a.md"])
        self.assertEqual(m["denominator"], 2)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["reciprocal_rank"], 0.0)
        self.assertEqual(m["average_precision"], 0.0)
        self.assertEqual(m["ndcg"], 0.0)

    def test_no_relevant_retrieved_still_counts_as_a_miss(self):
        m = metrics.ranking_metrics(["b", "c", "d"], ["a.md"])
        self.assertEqual(m["status"], "scored")
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["reciprocal_rank"], 0.0)  # a genuine miss, not None
        self.assertEqual(m["average_precision"], 0.0)
        self.assertEqual(m["ndcg"], 0.0)

    def test_empty_retrieval(self):
        m = metrics.ranking_metrics([], ["a.md"])
        self.assertEqual(m["status"], "empty_retrieval")
        self.assertIsNone(m["precision"])  # zero denominator -> unavailable
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["reciprocal_rank"], 0.0)
        self.assertEqual(m["average_precision"], 0.0)
        self.assertEqual(m["ndcg"], 0.0)

    def test_zero_relevant_ground_truth_is_null_not_zero(self):
        m = metrics.ranking_metrics(["a.md"], [])
        self.assertEqual(m["status"], "no_ground_truth")
        for key in ("precision", "recall", "reciprocal_rank", "average_precision", "ndcg"):
            self.assertIsNone(m[key], key)

    def test_missing_labels_are_null_not_zero(self):
        m = metrics.retrieval_metrics(_row(ground_truth_sources="", retrieved_sources="a.md"))
        self.assertEqual(m["status"], "no_ground_truth")
        self.assertIsNone(m["reciprocal_rank"])
        self.assertIsNone(m["average_precision"])
        self.assertIsNone(m["ndcg"])

    def test_missing_retrieval_data_is_null_not_zero(self):
        m = metrics.retrieval_metrics(
            _row(ground_truth_sources="a.md", retrieved_sources="", retrieval_available=False)
        )
        self.assertEqual(m["status"], "no_retrieval_data")
        self.assertIsNone(m["reciprocal_rank"])
        self.assertIsNone(m["average_precision"])
        self.assertIsNone(m["ndcg"])


# --------------------------------------------------------------------------- #
# Aggregate MRR / MAP@5 / nDCG@5 with sample counts and coverage
# --------------------------------------------------------------------------- #
class RankingAggregateTests(unittest.TestCase):
    def test_means_samples_and_coverage(self):
        rows = [
            _row(question_id="A", ground_truth_sources="a.md", retrieved_sources="a.md"),
            _row(question_id="B", ground_truth_sources="a.md", retrieved_sources="b.md"),
            _row(question_id="C", ground_truth_sources="", retrieved_sources="a.md"),
            _row(question_id="D", ground_truth_sources="a.md", retrieved_sources="", retrieval_available=False),
        ]
        ret = metrics.aggregate_system(rows)["automatic"]["retrieval"]
        self.assertAlmostEqual(ret["precision_at_5_mean"], 0.5, places=4)
        self.assertAlmostEqual(ret["recall_at_5_mean"], 0.5, places=4)
        self.assertAlmostEqual(ret["mrr"], 0.5, places=4)
        self.assertAlmostEqual(ret["map_at_5"], 0.5, places=4)
        self.assertAlmostEqual(ret["ndcg_at_5"], 0.5, places=4)
        self.assertEqual(ret["mrr_sample"], 2)  # A and B (D has no retrieval data)
        self.assertEqual(ret["labelled_questions"], 3)  # A, B, D
        self.assertEqual(ret["total_questions"], 4)
        self.assertEqual(ret["coverage"], 0.75)
        self.assertEqual(ret["relevance"], "binary")

    def test_aggregates_are_null_without_labels(self):
        rows = [_row(question_id="A", ground_truth_sources="", retrieved_sources="a.md")]
        ret = metrics.aggregate_system(rows)["automatic"]["retrieval"]
        self.assertIsNone(ret["mrr"])
        self.assertIsNone(ret["map_at_5"])
        self.assertIsNone(ret["ndcg_at_5"])
        self.assertEqual(ret["mrr_sample"], 0)
        self.assertEqual(ret["excluded_no_ground_truth"], 1)

    def test_metric_definitions_document_the_new_metrics(self):
        defs = metrics.metric_definitions()
        for key in ("retrieval_mrr", "retrieval_map_at_5", "retrieval_ndcg_at_5"):
            self.assertIn(key, defs)
            self.assertEqual(defs[key]["relevance"], "binary")


if __name__ == "__main__":
    unittest.main()
