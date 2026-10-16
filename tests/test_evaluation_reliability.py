"""Offline unit tests for evaluation reliability and metric correctness.

Standard-library ``unittest`` only, and no network calls: pure functions and
in-memory/temporary data. (``evaluation.run_evaluation`` imports its heavy
dependencies lazily, so importing it here does not touch Groq/Pinecone.)
"""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.exceptions import OutputParserException

from evaluation import manual_scores
from evaluation import metrics
from evaluation import run_evaluation as re


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
class ClassifyErrorTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(re.classify_error(OutputParserException("Failed to parse JSON")), re.ERROR_PARSER)

    def test_quota_by_message(self):
        self.assertEqual(
            re.classify_error(RuntimeError("Rate limit reached: TPD quota exhausted")), re.ERROR_QUOTA
        )

    def test_quota_by_type_name(self):
        class RateLimitError(Exception):
            pass

        self.assertEqual(re.classify_error(RateLimitError("too many requests")), re.ERROR_QUOTA)

    def test_application(self):
        self.assertEqual(re.classify_error(RuntimeError("unexpected internal bug")), re.ERROR_APPLICATION)

    def test_auth_and_network(self):
        class AuthenticationError(Exception):
            pass

        class ConnectionReset(Exception):
            pass

        self.assertEqual(re.classify_error(AuthenticationError("Invalid API Key")), re.ERROR_AUTH)
        self.assertEqual(re.classify_error(ConnectionReset("connection reset")), re.ERROR_NETWORK)


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #
class ResultSchemaTests(unittest.TestCase):
    def test_new_columns_present(self):
        for col in ("retrieval_available", "used_sources", "used_source_count", "error_category"):
            self.assertIn(col, re.RESULT_FIELDS)

    def test_blank_row_defaults(self):
        row = re._blank_row(
            "run", "baseline_rag",
            {"id": "Q1", "question": "q", "category": "direct", "relevant_sources": [], "expected_answer_points": []},
        )
        self.assertEqual(row["error_category"], "")
        self.assertFalse(row["retrieval_available"])
        self.assertEqual(row["used_sources"], "")
        self.assertEqual(row["used_source_count"], 0)


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def _rrow(**kwargs):
    base = {
        "ground_truth_sources": "",
        "retrieved_sources": "",
        "used_sources": "",
        "retrieval_available": True,
        "request_failed": False,
    }
    base.update(kwargs)
    return base


class RetrievalMetricTests(unittest.TestCase):
    def test_dedupe_by_source_document(self):
        # Three chunks of the same relevant doc -> one distinct document, P=1.0.
        m = metrics.retrieval_metrics(
            _rrow(ground_truth_sources="a.md", retrieved_sources="a.md;a.md;a.md")
        )
        self.assertEqual(m["status"], "scored")
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["denominator"], 1)  # not inflated to 3

    def test_precision_and_recall_multiple_docs(self):
        m = metrics.retrieval_metrics(
            _rrow(ground_truth_sources="a.md;c.md", retrieved_sources="a.md;b.md")
        )
        self.assertEqual(m["precision"], 0.5)  # 1 of 2 distinct retrieved docs
        self.assertEqual(m["recall"], 0.5)  # 1 of 2 gt docs

    def test_denominator_capped_at_5(self):
        # 6 distinct docs retrieved -> only the raw top-5 are evaluated.
        m = metrics.retrieval_metrics(
            _rrow(ground_truth_sources="a.md", retrieved_sources="a.md;b;c;d;e;f")
        )
        self.assertEqual(m["denominator"], 5)
        self.assertEqual(m["precision"], 0.2)

    def test_empty_retrieval(self):
        m = metrics.retrieval_metrics(_rrow(ground_truth_sources="a.md", retrieved_sources=""))
        self.assertEqual(m["status"], "empty_retrieval")
        self.assertIsNone(m["precision"])  # zero denominator -> unavailable, not 0
        self.assertEqual(m["recall"], 0.0)  # genuine miss

    def test_missing_ground_truth_excluded(self):
        m = metrics.retrieval_metrics(_rrow(ground_truth_sources="", retrieved_sources="a.md"))
        self.assertEqual(m["status"], "no_ground_truth")
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["recall"])

    def test_missing_retrieval_data_is_unavailable_not_zero(self):
        m = metrics.retrieval_metrics(
            _rrow(ground_truth_sources="a.md", retrieved_sources="", retrieval_available=False)
        )
        self.assertEqual(m["status"], "no_retrieval_data")
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["recall"])

    def test_used_sources_not_scored(self):
        # Retrieval metrics must ignore documents used after grading/rewrite/web.
        m = metrics.retrieval_metrics(
            _rrow(ground_truth_sources="a.md", retrieved_sources="", used_sources="a.md")
        )
        self.assertEqual(m["status"], "empty_retrieval")

    def test_backcompat_missing_availability_column(self):
        row = {"ground_truth_sources": "a.md", "retrieved_sources": "a.md"}
        m = metrics.retrieval_metrics(row)  # no retrieval_available key
        self.assertEqual(m["status"], "scored")


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
class PercentileTests(unittest.TestCase):
    def test_linear_interpolation(self):
        self.assertIsNone(metrics.percentile([], 95))
        self.assertEqual(metrics.percentile([10.0], 95), 10.0)
        self.assertAlmostEqual(metrics.percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)

    def test_latency_stats_p50_is_median(self):
        stats = metrics._latency_stats([10.0, 20.0, 30.0])
        self.assertEqual(stats["p50"], 20.0)
        self.assertAlmostEqual(stats["p95"], 29.0)

    def test_latency_stats_empty(self):
        stats = metrics._latency_stats([])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["p50"])
        self.assertIsNone(stats["p95"])


# --------------------------------------------------------------------------- #
# Aggregation: reliability, latency separation, manual, citation
# --------------------------------------------------------------------------- #
def _arow(**kwargs):
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


class AggregateTests(unittest.TestCase):
    def test_reliability_rates_and_categories(self):
        rows = [
            _arow(question_id="A"),
            _arow(question_id="B", request_failed="True", error_category="quota_or_rate_limit"),
            _arow(question_id="C", request_failed="True", error_category="parser"),
            _arow(question_id="D", request_failed="True", error_category="quota_or_rate_limit"),
        ]
        auto = metrics.aggregate_system(rows)["automatic"]
        self.assertEqual(auto["questions_total"], 4)
        self.assertEqual(auto["requests_ok"], 1)
        self.assertEqual(auto["request_failures"], 3)
        self.assertEqual(auto["success_rate"], 0.25)
        self.assertEqual(auto["failure_rate"], 0.75)
        self.assertEqual(auto["failure_categories"], {"quota_or_rate_limit": 2, "parser": 1})

    def test_latency_success_separated_from_fast_failures(self):
        rows = [
            _arow(question_id="A", latency_seconds="10.0"),
            _arow(question_id="B", latency_seconds="20.0"),
            _arow(question_id="C", latency_seconds="30.0"),
            _arow(question_id="D", request_failed="True", error_category="quota_or_rate_limit", latency_seconds="0.1"),
            _arow(question_id="E", request_failed="True", error_category="quota_or_rate_limit", latency_seconds="0.2"),
        ]
        lat = metrics.aggregate_system(rows)["automatic"]["latency_seconds"]
        self.assertEqual(lat["successful"]["n"], 3)
        self.assertEqual(lat["successful"]["p50"], 20.0)  # failures excluded
        self.assertAlmostEqual(lat["successful"]["p95"], 29.0)
        self.assertEqual(lat["failed"]["n"], 2)
        self.assertAlmostEqual(lat["failed"]["p50"], 0.15)

    def test_retrieval_sample_counts_and_exclusions(self):
        rows = [
            _arow(question_id="A", ground_truth_sources="a.md", retrieved_sources="a.md"),
            _arow(question_id="B", ground_truth_sources="a.md", retrieved_sources=""),  # empty
            _arow(question_id="C", ground_truth_sources=""),  # no gt
            _arow(question_id="D", ground_truth_sources="a.md", retrieved_sources="", retrieval_available=False),
        ]
        ret = metrics.aggregate_system(rows)["automatic"]["retrieval"]
        self.assertEqual(ret["precision_sample"], 1)
        self.assertEqual(ret["recall_sample"], 2)  # scored + empty
        self.assertEqual(ret["excluded_no_ground_truth"], 1)
        self.assertEqual(ret["excluded_no_retrieval_data"], 1)
        self.assertEqual(ret["empty_retrieval"], 1)
        self.assertEqual(ret["precision_at_5_mean"], 1.0)

    def test_retrieval_includes_failed_requests_when_raw_data_exists(self):
        # A downstream quota failure must not remove valid raw retrieval data.
        rows = [
            _arow(question_id="A", ground_truth_sources="a.md", retrieved_sources="a.md",
                  request_failed="True", error_category="quota_or_rate_limit"),
        ]
        ret = metrics.aggregate_system(rows)["automatic"]["retrieval"]
        self.assertEqual(ret["precision_sample"], 1)
        self.assertEqual(ret["precision_at_5_mean"], 1.0)

    def test_manual_missing_then_scored(self):
        blank = metrics.aggregate_system([_arow(question_id="A")])["manual"]
        self.assertEqual(blank["answer_correctness"]["n"], 0)
        self.assertIsNone(blank["answer_correctness"]["mean"])
        self.assertEqual(blank["status"], "not_yet_reviewed")

        scored = metrics.aggregate_system(
            [
                _arow(question_id="A", answer_correctness_manual="5", faithfulness_manual="4"),
                _arow(question_id="B", answer_correctness_manual="3"),
            ]
        )["manual"]
        self.assertEqual(scored["answer_correctness"]["n"], 2)
        self.assertEqual(scored["answer_correctness"]["mean"], 4.0)
        self.assertEqual(scored["faithfulness"]["mean"], 4.0)
        self.assertEqual(scored["status"], "scored")

    def test_manual_failed_rows_excluded(self):
        rows = [
            _arow(question_id="A", answer_correctness_manual="5"),
            _arow(question_id="B", request_failed="True", error_category="parser", answer_correctness_manual="5"),
        ]
        manual = metrics.aggregate_system(rows)["manual"]
        self.assertEqual(manual["answer_correctness"]["n"], 1)
        self.assertEqual(manual["rows_excluded_as_failed"], 1)

    def test_manual_out_of_range_not_averaged(self):
        manual = metrics.aggregate_system([_arow(question_id="A", answer_correctness_manual="9")])["manual"]
        self.assertEqual(manual["answer_correctness"]["n"], 0)
        self.assertEqual(manual["out_of_range_values"], 1)

    def test_citation_metrics_not_yet_available(self):
        cit = metrics.aggregate_system([_arow()])["manual"]["citation_quality"]
        self.assertEqual(cit["status"], "not_yet_available")
        self.assertIsNone(cit["citation_precision"])
        self.assertIsNone(cit["citation_recall"])
        self.assertTrue(cit["required_annotations"])

    def test_token_usage_only_when_returned(self):
        rows = [
            _arow(token_usage_available="True", prompt_tokens="100", completion_tokens="20", total_tokens="120"),
            _arow(request_failed="True", error_category="parser"),
        ]
        tokens = metrics.aggregate_system(rows)["automatic"]["tokens"]
        self.assertTrue(tokens["available"])
        self.assertEqual(tokens["rows_with_usage"], 1)
        self.assertEqual(tokens["total_tokens_sum"], 120)


# --------------------------------------------------------------------------- #
# Manual score preservation
# --------------------------------------------------------------------------- #
class ManualScorePersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.store = self.tmp / manual_scores.MANUAL_SCORES_FILENAME

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_preserves_scores(self):
        scored = [
            _arow(run_id="R1", question_id="A", answer_correctness_manual="5", faithfulness_manual="4",
                  manual_notes="good"),
            _arow(run_id="R1", question_id="B", answer_correctness_manual="", faithfulness_manual=""),
        ]
        n = manual_scores.persist_manual_scores(self.store, scored)
        self.assertEqual(n, 1)

        # A freshly regenerated file with blank manual columns.
        blank = [_arow(run_id="R1", question_id="A"), _arow(run_id="R1", question_id="B")]
        filled = manual_scores.apply_manual_scores(blank, manual_scores.load_manual_scores(self.store))
        self.assertEqual(filled, 1)
        self.assertEqual(blank[0]["answer_correctness_manual"], "5")
        self.assertEqual(blank[0]["faithfulness_manual"], "4")
        self.assertEqual(blank[1]["answer_correctness_manual"], "")

    def test_persist_keeps_other_runs(self):
        manual_scores.persist_manual_scores(
            self.store, [_arow(run_id="R1", question_id="A", answer_correctness_manual="5")]
        )
        # A later persist for a different run must not drop R1.
        manual_scores.persist_manual_scores(
            self.store, [_arow(run_id="R2", question_id="A", answer_correctness_manual="3")]
        )
        store = manual_scores.load_manual_scores(self.store)
        self.assertIn(("R1", "baseline_rag", "A"), store)
        self.assertIn(("R2", "baseline_rag", "A"), store)

    def test_no_scores_means_no_file(self):
        n = manual_scores.persist_manual_scores(self.store, [_arow(question_id="A")])
        self.assertEqual(n, 0)
        self.assertFalse(self.store.exists())

    def test_apply_does_not_overwrite_existing(self):
        store = {("R1", "baseline_rag", "A"): {"answer_correctness_manual": "5",
                                               "faithfulness_manual": "", "manual_notes": ""}}
        rows = [_arow(run_id="R1", question_id="A", answer_correctness_manual="2")]
        manual_scores.apply_manual_scores(rows, store)
        self.assertEqual(rows[0]["answer_correctness_manual"], "2")  # row wins


# --------------------------------------------------------------------------- #
# Latest-run selection
# --------------------------------------------------------------------------- #
class NewestResultsFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_latest_copy_is_ignored(self):
        canonical = self.tmp / "per_question_20200101T000000Z.csv"
        latest = self.tmp / "per_question_latest.csv"
        canonical.write_text("run_id\n", encoding="utf-8")
        latest.write_text("run_id\n", encoding="utf-8")
        # Make the convenience copy the most recently modified file.
        os.utime(canonical, (1000, 1000))
        os.utime(latest, (2000, 2000))
        self.assertEqual(metrics.newest_results_file(self.tmp), canonical)

    def test_no_candidates(self):
        self.assertIsNone(metrics.newest_results_file(self.tmp))


# --------------------------------------------------------------------------- #
# Summary CSV / JSON consistency
# --------------------------------------------------------------------------- #
class SummaryConsistencyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_results(self, rows):
        path = self.tmp / "per_question_RUNX.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=re.RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_summary_json_and_csv_agree(self):
        rows = [
            _full_row("RUNX", "baseline_rag", "Q1", ground_truth_sources="a.md", retrieved_sources="a.md",
                      latency_seconds="2.0", answer_correctness_manual="5"),
            _full_row("RUNX", "baseline_rag", "Q2", ground_truth_sources="a.md", retrieved_sources="a.md",
                      latency_seconds="4.0"),
            _full_row("RUNX", "axiom_self_rag", "Q1", ground_truth_sources="a.md", retrieved_sources="a.md",
                      latency_seconds="3.0", request_failed="True", error_category="quota_or_rate_limit"),
        ]
        path = self._write_results(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = metrics.main(["--results", str(path), "--out-dir", str(self.tmp)])
        self.assertEqual(rc, 0)

        summary = json.loads((self.tmp / "summary_RUNX.json").read_text(encoding="utf-8"))
        with (self.tmp / "summary_RUNX.csv").open("r", encoding="utf-8", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))

        self.assertIn("metric_definitions", summary)
        self.assertEqual(sorted(summary["systems"].keys()), ["axiom_self_rag", "baseline_rag"])
        self.assertEqual(len(csv_rows), 2)

        csv_by_system = {r["system_version"]: r for r in csv_rows}
        for system, blocks in summary["systems"].items():
            auto = blocks["automatic"]
            self.assertEqual(float(csv_by_system[system]["success_rate"]), auto["success_rate"])
            self.assertEqual(
                float(csv_by_system[system]["retrieval_precision_at_5_mean"]),
                auto["retrieval"]["precision_at_5_mean"],
            )
            self.assertEqual(csv_by_system[system]["citation_precision_status"], "not_yet_available")

    def test_unavailable_metrics_are_null_not_zero(self):
        rows = [
            _full_row("RUNY", "baseline_rag", "Q1", ground_truth_sources="", retrieved_sources="a.md",
                      latency_seconds="1.0"),
        ]
        path = self._write_results(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics.main(["--results", str(path), "--out-dir", str(self.tmp)])
        summary = json.loads((self.tmp / "summary_RUNY.json").read_text(encoding="utf-8"))
        ret = summary["systems"]["baseline_rag"]["automatic"]["retrieval"]
        self.assertIsNone(ret["precision_at_5_mean"])
        self.assertIsNone(ret["recall_at_5_mean"])
        self.assertEqual(ret["excluded_no_ground_truth"], 1)


def _full_row(run_id, system, qid, **kwargs):
    row = re._blank_row(
        run_id, system,
        {"id": qid, "question": f"q-{qid}", "category": "direct",
         "relevant_sources": [], "expected_answer_points": ["p"]},
    )
    row.update(
        {
            "ground_truth_sources": "",
            "retrieved_sources": "",
            "retrieval_available": True,
            "latency_seconds": "1.0",
        }
    )
    row.update(kwargs)
    return row


if __name__ == "__main__":
    unittest.main()
