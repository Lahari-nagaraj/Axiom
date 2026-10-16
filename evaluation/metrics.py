"""Aggregate evaluation metrics from per-question results.

Reads the per-question CSV (or JSON) produced by ``evaluation/run_evaluation.py``
and writes an aggregate summary as JSON and CSV.

Two clearly separated metric groups are reported:

* AUTOMATIC measurements computed directly from the run: retrieval
  Precision@5 / Recall@5 over the shared raw top-k retrieval, latency (P50/P95,
  successful requests reported separately from failed ones), request
  reliability (success/failure rates and failure categories), and token usage
  *when the provider actually returned it*.
* MANUAL scores read from the human-editable columns
  (``answer_correctness_manual``, ``faithfulness_manual``) plus citation
  metrics, which are reported as *not yet available* because the data does not
  yet contain per-claim citation annotations. Nothing is fabricated.

Usage (from the project root)::

    python evaluation/metrics.py
    python evaluation/metrics.py --results evaluation/results/per_question_<runid>.csv

If ``--results`` is omitted, the newest canonical ``per_question_*.csv`` in
``evaluation/results/`` is used (the ``per_question_latest.csv`` convenience copy
is never chosen).

Manual scores are merged from ``evaluation/results/manual_scores.csv`` (keyed by
``run_id``, ``system_version``, ``question_id``) so reviewed scores survive
regeneration of the per-question files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # package import (python -m evaluation.metrics)
    from evaluation.manual_scores import (
        MANUAL_SCORES_FILENAME,
        apply_manual_scores,
        collect_manual_scores,
        load_manual_scores,
        persist_manual_scores,
    )
except ImportError:  # standalone script (python evaluation\metrics.py)
    from manual_scores import (  # type: ignore
        MANUAL_SCORES_FILENAME,
        apply_manual_scores,
        collect_manual_scores,
        load_manual_scores,
        persist_manual_scores,
    )

EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"
LATEST_RESULTS_NAME = "per_question_latest.csv"
RETRIEVAL_K = 5
MANUAL_MIN, MANUAL_MAX = 1, 5


# --------------------------------------------------------------------------- #
# Small parsing helpers (never invent values)
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "n/a", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "n/a", "none", "null"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _split_sources(value: Any) -> List[str]:
    if not value:
        return []
    return [s.strip() for s in str(value).split(";") if s.strip()]


def _distinct(items: List[str]) -> List[str]:
    """Deduplicate while preserving order (used to dedupe by source document)."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _retrieval_available(row: Dict[str, Any]) -> bool:
    """Whether raw top-k retrieval data was recorded for this row.

    Files written before the ``retrieval_available`` column existed always
    recorded raw retrieval, so they default to True.
    """
    if "retrieval_available" in row:
        return _to_bool(row.get("retrieval_available"))
    return True


def percentile(values: List[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile (same definition as numpy.percentile).

    Uses rank = (n - 1) * p/100 and interpolates between the two neighbouring
    order statistics. Returns None for an empty sample (never 0).
    """
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 4)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return round(ordered[int(rank)], 4)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo), 4)


def _latency_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": round(statistics.median(values), 4),  # P50 = median response time
        "p95": percentile(values, 95),  # linear interpolation
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def retrieval_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieval Precision@5 / Recall@5 over the shared raw top-k retrieval.

    * Retrieved chunks are sliced to the raw top-5, then **deduplicated by
      source document** so several chunks from one document cannot inflate
      precision. The precision denominator is therefore the number of distinct
      retrieved documents (at most 5).
    * Precision@5 = (distinct retrieved documents that are in ground truth) /
      (distinct retrieved documents).
    * Recall@5 = (distinct ground-truth documents present in the top-5) /
      (total distinct ground-truth documents).
    * This scores raw retrieval only. Documents used after Self-RAG grading,
      query rewriting or web search live in ``used_sources`` and are never
      compared against runbook filenames.

    ``status`` explains any missing value so callers never substitute zero:

    * ``no_ground_truth``    - ambiguous/unanswerable question (both None)
    * ``no_retrieval_data``  - raw retrieval was not recorded (both None)
    * ``empty_retrieval``    - retrieval ran but returned nothing
                               (precision None, recall 0.0 - a genuine miss)
    * ``scored``             - both metrics computed
    """
    gt_list = _distinct(_split_sources(row.get("ground_truth_sources")))
    base = {
        "precision": None,
        "recall": None,
        "status": "no_ground_truth",
        "denominator": 0,
        "gt_count": len(gt_list),
        "retrieved_count": 0,
        "retrieved_raw_count": 0,
    }
    if not gt_list:
        return base
    if not _retrieval_available(row):
        return {**base, "status": "no_retrieval_data"}

    retrieved_raw = _split_sources(row.get("retrieved_sources"))[:RETRIEVAL_K]
    retrieved = _distinct(retrieved_raw)
    base["retrieved_raw_count"] = len(retrieved_raw)
    base["retrieved_count"] = len(retrieved)
    if not retrieved:
        return {**base, "status": "empty_retrieval", "recall": 0.0}

    gt_set = set(gt_list)
    hits = [s for s in retrieved if s in gt_set]
    base.update(
        {
            "precision": round(len(hits) / len(retrieved), 4),
            "recall": round(len(set(hits)) / len(gt_list), 4),
            "status": "scored",
            "denominator": len(retrieved),
        }
    )
    return base


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_system(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    failed_rows = [r for r in rows if _to_bool(r.get("request_failed"))]
    ok_rows = [r for r in rows if not _to_bool(r.get("request_failed"))]
    failures = len(failed_rows)

    # --- Reliability -------------------------------------------------------
    failure_categories: Dict[str, int] = {}
    for r in failed_rows:
        cat = str(r.get("error_category") or "unknown")
        failure_categories[cat] = failure_categories.get(cat, 0) + 1

    # --- Latency: successful and failed requests kept separate. Fast quota
    # failures must never make successful responses look faster. -------------
    success_lat = [
        v for v in (_to_float(r.get("latency_seconds")) for r in ok_rows) if v is not None
    ]
    failed_lat = [
        v for v in (_to_float(r.get("latency_seconds")) for r in failed_rows) if v is not None
    ]

    # --- Token usage: only rows where the provider returned numbers. -------
    prompt_vals = [v for v in (_to_int(r.get("prompt_tokens")) for r in rows) if v is not None]
    completion_vals = [v for v in (_to_int(r.get("completion_tokens")) for r in rows) if v is not None]
    total_vals = [v for v in (_to_int(r.get("total_tokens")) for r in rows) if v is not None]
    usage_rows = sum(1 for r in rows if _to_bool(r.get("token_usage_available")))

    # --- Retrieval: raw top-k shared by both systems, deduped by document. --
    precisions, recalls = [], []
    status_counts: Dict[str, int] = {}
    for r in rows:
        m = retrieval_metrics(r)
        status_counts[m["status"]] = status_counts.get(m["status"], 0) + 1
        if m["precision"] is not None:
            precisions.append(m["precision"])
        if m["recall"] is not None:
            recalls.append(m["recall"])

    # --- Manual scores: only reviewed AND successful rows. -----------------
    correctness, faithfulness = [], []
    out_of_range = 0
    excluded_failed_manual = 0
    for r in rows:
        if _to_bool(r.get("request_failed")):
            if _to_int(r.get("answer_correctness_manual")) is not None or _to_int(
                r.get("faithfulness_manual")
            ) is not None:
                excluded_failed_manual += 1
            continue
        c = _to_int(r.get("answer_correctness_manual"))
        if c is not None:
            if MANUAL_MIN <= c <= MANUAL_MAX:
                correctness.append(c)
            else:
                out_of_range += 1
        f = _to_int(r.get("faithfulness_manual"))
        if f is not None:
            if MANUAL_MIN <= f <= MANUAL_MAX:
                faithfulness.append(f)
            else:
                out_of_range += 1

    def _manual_stats(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {"mean": None, "n": 0, "min": None, "max": None}
        return {
            "mean": round(statistics.fmean(vals), 4),
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    per_category: Dict[str, Any] = {}
    for r in rows:
        cat = str(r.get("category") or "unknown")
        bucket = per_category.setdefault(cat, {"count": 0, "failures": 0})
        bucket["count"] += 1
        if _to_bool(r.get("request_failed")):
            bucket["failures"] += 1

    return {
        "automatic": {
            "questions_total": total,
            "requests_ok": len(ok_rows),
            "request_failures": failures,
            "success_rate": round(len(ok_rows) / total, 4) if total else None,
            "failure_rate": round(failures / total, 4) if total else None,
            "failure_categories": failure_categories,
            "latency_seconds": {
                "definition": (
                    "P50 = median response time; P95 = linear-interpolation percentile "
                    "(rank = (n-1)*p/100); successful and failed requests reported separately."
                ),
                "successful": _latency_stats(success_lat),
                "failed": _latency_stats(failed_lat),
            },
            "tokens": {
                "available": usage_rows > 0,
                "rows_with_usage": usage_rows,
                "prompt_tokens_sum": sum(prompt_vals) if prompt_vals else None,
                "completion_tokens_sum": sum(completion_vals) if completion_vals else None,
                "total_tokens_sum": sum(total_vals) if total_vals else None,
                "avg_total_tokens_per_usage_row": (
                    round(statistics.fmean(total_vals), 2) if total_vals else None
                ),
            },
            "retrieval": {
                "unit": "document-level, over the shared raw top-k retrieval",
                "k": RETRIEVAL_K,
                "dedup": "retrieved chunks are deduplicated by source document before scoring",
                "precision_at_5_mean": round(statistics.fmean(precisions), 4) if precisions else None,
                "recall_at_5_mean": round(statistics.fmean(recalls), 4) if recalls else None,
                "precision_sample": len(precisions),
                "recall_sample": len(recalls),
                "status_counts": status_counts,
                "excluded_no_ground_truth": status_counts.get("no_ground_truth", 0),
                "excluded_no_retrieval_data": status_counts.get("no_retrieval_data", 0),
                "empty_retrieval": status_counts.get("empty_retrieval", 0),
            },
        },
        "manual": {
            "scale": f"{MANUAL_MIN}-{MANUAL_MAX}",
            "answer_correctness": {
                "label": f"Answer Correctness (manual, {MANUAL_MIN}-{MANUAL_MAX})",
                **_manual_stats(correctness),
            },
            "faithfulness": {
                "label": f"Faithfulness (manual, {MANUAL_MIN}-{MANUAL_MAX})",
                **_manual_stats(faithfulness),
            },
            "scored_questions": len(correctness),
            "out_of_range_values": out_of_range,
            "rows_excluded_as_failed": excluded_failed_manual,
            "status": "scored" if correctness or faithfulness else "not_yet_reviewed",
            "citation_quality": {
                "status": "not_yet_available",
                "citation_precision": None,
                "citation_recall": None,
                "definitions": {
                    "citation_precision": "supported cited claims / all cited claims",
                    "citation_recall": (
                        "required evidence items covered by citations / all required evidence items"
                    ),
                },
                "zero_denominator_rule": (
                    "If a denominator is 0 the metric is reported as unavailable (null), never 0."
                ),
                "required_annotations": [
                    "Per answer: the number of cited claims and how many are supported by the cited evidence.",
                    "Per question: the list of required evidence items and which are covered by citations.",
                ],
                "note": (
                    "The results contain no per-claim citation annotations, so these are "
                    "reported as unavailable (null) rather than zero. No LLM judge is used."
                ),
            },
        },
        "per_category": per_category,
    }


# --------------------------------------------------------------------------- #
# Metric definitions (included in the summary so it is self-describing)
# --------------------------------------------------------------------------- #
def metric_definitions() -> Dict[str, Any]:
    return {
        "retrieval_precision_at_5": {
            "label": "Retrieval Precision@5",
            "formula": (
                "(# distinct retrieved documents in the top-5 that are in ground truth) / "
                "(# distinct retrieved documents in the top-5)"
            ),
            "denominator_cap": RETRIEVAL_K,
            "dedup": "deduplicated by source document",
            "scope": "shared raw top-k retrieval; excludes post-grading/rewrite/web 'used' documents",
            "excluded": "questions with no ground-truth sources, or with no recorded raw retrieval",
        },
        "retrieval_recall_at_5": {
            "label": "Retrieval Recall@5",
            "formula": (
                "(# distinct ground-truth documents present in the top-5) / "
                "(total # distinct ground-truth documents)"
            ),
            "scope": "shared raw top-k retrieval",
            "excluded": "questions with no ground-truth sources",
        },
        "answer_correctness_manual": {
            "label": f"Answer Correctness (manual, {MANUAL_MIN}-{MANUAL_MAX})",
            "formula": "mean of reviewed scores (1-5); n = number of reviewed successful answers",
            "rubric": "1 = incorrect, 2 = mostly incorrect, 3 = partially correct, 4 = mostly correct, 5 = fully correct",
            "excluded": "failed requests; out-of-range values are counted but not averaged",
        },
        "faithfulness_manual": {
            "label": f"Faithfulness (manual, {MANUAL_MIN}-{MANUAL_MAX})",
            "formula": "mean of reviewed scores (1-5)",
            "rubric": "1 = unsupported/contradicts evidence, 3 = mixed, 5 = fully grounded in cited sources",
            "excluded": "failed requests",
        },
        "citation_precision": {
            "label": "Citation Precision",
            "formula": "supported cited claims / all cited claims",
            "status": "not_yet_available",
            "required_annotations": "per-answer cited-claim support counts",
        },
        "citation_recall": {
            "label": "Citation Recall",
            "formula": "required evidence items covered by citations / all required evidence items",
            "status": "not_yet_available",
            "required_annotations": "per-question required-evidence coverage annotation",
        },
        "latency_p50_p95": {
            "label": "Latency P50 / P95",
            "formula": "P50 = median; P95 = linear-interpolation percentile of per-question wall time",
            "scope": "successful and failed requests reported separately",
        },
        "success_rate": {"label": "Success Rate", "formula": "successful requests / total requests"},
        "failure_rate": {"label": "Failure Rate", "formula": "failed requests / total requests"},
        "failure_categories": {
            "label": "Failure Categories",
            "formula": "count of failed requests per error_category",
            "categories": ["quota_or_rate_limit", "parser", "auth", "network", "application", "unknown"],
        },
        "token_usage": {
            "label": "Token Usage",
            "formula": "sum of provider-reported tokens; unavailable (null) when the provider returns none",
        },
    }


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def newest_results_file(results_dir: Path) -> Optional[Path]:
    """Newest canonical run file; the ``per_question_latest.csv`` copy is ignored."""
    candidates = [
        p
        for p in results_dir.glob("per_question_*.csv")
        if p.name != LATEST_RESULTS_NAME
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for system, blocks in summary["systems"].items():
        auto, manual = blocks["automatic"], blocks["manual"]
        ret = auto["retrieval"]
        lat = auto["latency_seconds"]
        cit = manual["citation_quality"]
        out.append(
            {
                "run_id": summary["run_id"],
                "system_version": system,
                "questions_total": auto["questions_total"],
                "requests_ok": auto["requests_ok"],
                "request_failures": auto["request_failures"],
                "success_rate": auto["success_rate"],
                "failure_rate": auto["failure_rate"],
                "failure_categories": json.dumps(auto.get("failure_categories", {}), sort_keys=True),
                "latency_success_n": lat["successful"]["n"],
                "latency_success_mean_s": lat["successful"]["mean"],
                "latency_success_p50_s": lat["successful"]["p50"],
                "latency_success_p95_s": lat["successful"]["p95"],
                "latency_failed_n": lat["failed"]["n"],
                "latency_failed_mean_s": lat["failed"]["mean"],
                "latency_failed_p50_s": lat["failed"]["p50"],
                "latency_failed_p95_s": lat["failed"]["p95"],
                "retrieval_precision_at_5_mean": ret["precision_at_5_mean"],
                "retrieval_recall_at_5_mean": ret["recall_at_5_mean"],
                "retrieval_precision_sample": ret["precision_sample"],
                "retrieval_recall_sample": ret["recall_sample"],
                "retrieval_excluded_no_ground_truth": ret["excluded_no_ground_truth"],
                "retrieval_excluded_no_retrieval_data": ret["excluded_no_retrieval_data"],
                "retrieval_empty_retrieval": ret["empty_retrieval"],
                "token_usage_available": auto["tokens"]["available"],
                "total_tokens_sum": auto["tokens"]["total_tokens_sum"],
                "answer_correctness_manual_mean": manual["answer_correctness"]["mean"],
                "answer_correctness_manual_n": manual["answer_correctness"]["n"],
                "faithfulness_manual_mean": manual["faithfulness"]["mean"],
                "faithfulness_manual_n": manual["faithfulness"]["n"],
                "manual_rows_excluded_as_failed": manual["rows_excluded_as_failed"],
                "manual_out_of_range_values": manual["out_of_range_values"],
                "citation_precision_status": cit["status"],
                "citation_recall_status": cit["status"],
                "manual_status": manual["status"],
            }
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Axiom evaluation results.")
    parser.add_argument("--results", default="", help="Path to per_question_*.csv (default: newest canonical run).")
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for summary files.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    results_path = Path(args.results) if args.results else newest_results_file(out_dir)
    if not results_path or not results_path.exists():
        print("ERROR: no results file found. Run evaluation/run_evaluation.py first.")
        return 2

    rows = load_rows(results_path)
    if not rows:
        print(f"ERROR: results file is empty: {results_path}")
        return 2

    # Merge durable manual scores so reviewed values survive regeneration. The
    # per-question file's own scores win; the store fills any blanks.
    manual_path = out_dir / MANUAL_SCORES_FILENAME
    store = load_manual_scores(manual_path)
    merged = {**store, **collect_manual_scores(rows)}
    apply_manual_scores(rows, merged)
    persist_manual_scores(manual_path, rows)

    run_id = str(rows[0].get("run_id") or "unknown")
    systems = sorted({str(r.get("system_version") or "unknown") for r in rows})

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_results_file": str(results_path),
        "metric_definitions": metric_definitions(),
        "legend": {
            "automatic": "Measured directly from the run (no human input).",
            "manual": "Human-scored fields; null until a reviewer fills them in.",
            "null": "Reported as null when a metric is unavailable or has a zero denominator - never 0.",
        },
        "systems": {},
    }
    for system in systems:
        summary["systems"][system] = aggregate_system(
            [r for r in rows if r.get("system_version") == system]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"summary_{run_id}.json"
    csv_path = out_dir / f"summary_{run_id}.csv"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(summary_rows(summary)[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows(summary))

    # --- Console report ---
    print("=" * 70)
    print(f"Axiom evaluation summary  (run {run_id})")
    print("=" * 70)
    for system, blocks in summary["systems"].items():
        auto, manual = blocks["automatic"], blocks["manual"]
        lat = auto["latency_seconds"]
        ret = auto["retrieval"]
        print(f"\n[{system}]")
        print("  AUTOMATIC")
        print(f"    questions            : {auto['questions_total']}")
        print(f"    ok / failed          : {auto['requests_ok']} / {auto['request_failures']}"
              f"  (success {auto['success_rate']}, failure {auto['failure_rate']})")
        if auto.get("failure_categories"):
            print(f"    failure categories   : {auto['failure_categories']}")
        print(f"    latency success      : n={lat['successful']['n']} "
              f"p50={lat['successful']['p50']} p95={lat['successful']['p95']}s")
        print(f"    latency failed       : n={lat['failed']['n']} "
              f"p50={lat['failed']['p50']} p95={lat['failed']['p95']}s")
        print(f"    retrieval P@5 / R@5  : {ret['precision_at_5_mean']} / {ret['recall_at_5_mean']} "
              f"(precision n={ret['precision_sample']}, recall n={ret['recall_sample']})")
        print(f"    retrieval exclusions : no_gt={ret['excluded_no_ground_truth']} "
              f"no_retrieval={ret['excluded_no_retrieval_data']} empty={ret['empty_retrieval']}")
        tokens = auto["tokens"]
        if tokens["available"]:
            print(f"    tokens (rows {tokens['rows_with_usage']}): total sum {tokens['total_tokens_sum']}")
        else:
            print("    tokens               : not returned by provider (recorded as blank)")
        print("  MANUAL (human-scored)")
        if manual["status"] == "not_yet_reviewed":
            print("    not yet reviewed - manual columns are blank")
        else:
            print(f"    answer correctness   : mean {manual['answer_correctness']['mean']} "
                  f"(n={manual['answer_correctness']['n']})")
            print(f"    faithfulness         : mean {manual['faithfulness']['mean']} "
                  f"(n={manual['faithfulness']['n']})")
        print(f"    citation P/R         : {manual['citation_quality']['status']}")

    print("-" * 70)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    if merged:
        print(f"Manual score store       : {manual_path} ({len(merged)} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
