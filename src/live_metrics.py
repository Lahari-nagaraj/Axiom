"""Live evaluation built from real execution artifacts (no manual labels).

The live dashboard is question-scoped. For every real request this module reads
the persisted live records and produces an **automatic** evaluation from what
actually happened:

* the original question,
* the chunks each system retrieved (source, rank, text and score when the
  retriever exposes one),
* the evidence a system actually used (Self-RAG keeps its final, post-grading
  evidence separate from its initial retrieval),
* the generated answer,
* the recorded Self-RAG verification decisions (IsSUP / IsUSE), retries and
  web-search activity.

The measurements come from :mod:`src.auto_eval` - deterministic, locally
computable lexical proxies plus the reported verification decisions. There are no
hard-coded questions, no ground-truth labels and no fabricated numbers.

Conventional IR metrics (Precision@5, Recall@5, MRR, MAP@5, nDCG@5) require a
relevance reference the live system does not have automatically, so they are
reported as ``unavailable`` with an explicit reason instead of a score. Retrieval
and answer measurements that cannot be computed are reported with an explicit
status (``unavailable`` / ``insufficient_data``) and a ``null`` value - never as
zero.

Results are persisted per ``(request_id, system_version, method)`` in
``live_auto_evaluations`` so opening a question's details reuses the stored
evaluation, never re-runs answer generation and never creates duplicates.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from evaluation import metrics as em
from src import auto_eval, live_db

SELF_RAG_SYSTEM = "axiom_self_rag"
BASELINE_SYSTEM = "baseline_rag"

SYSTEM_VIEW = {
    BASELINE_SYSTEM: "initial_retrieval",
    SELF_RAG_SYSTEM: "final_evidence",
}

LEGEND = {
    "ok": "Measured from persisted live records.",
    "unavailable": "The required input was not recorded (e.g. chunk text or a relevance reference).",
    "insufficient_data": "Not enough recorded data to compute this.",
    "null": "Reported as null when unavailable - never 0.",
}

AUTOMATIC_NOTE = (
    "Automatic proxy evaluation computed locally from the recorded question, "
    "retrieved chunks, final evidence and answer. No ground-truth labels, no "
    "hard-coded questions and no model call."
)


# --------------------------------------------------------------------------- #
# Primitive helpers
# --------------------------------------------------------------------------- #
def _latency(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "average_seconds": None,
            "p50_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "sample": 0,
            "status": "insufficient_data",
        }
    return {
        "average_seconds": round(statistics.fmean(values), 4),
        "p50_seconds": round(statistics.median(values), 4),
        "p95_seconds": em.percentile(values, 95),
        "min_seconds": round(min(values), 4),
        "max_seconds": round(max(values), 4),
        "sample": len(values),
        "status": "ok",
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _counts(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(requests)
    failed = [r for r in requests if r.get("status") != "ok"]
    ok = total - len(failed)
    categories: Dict[str, int] = {}
    for r in failed:
        cat = str(r.get("error_category") or "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    return {
        "total": total,
        "success": ok,
        "failed": len(failed),
        "success_rate": _rate(ok, total),
        "failure_rate": _rate(len(failed), total),
        "failure_categories": categories,
        "status": "ok" if total else "insufficient_data",
    }


def _tokens(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = [e["prompt_tokens"] for e in evaluations if e.get("prompt_tokens") is not None]
    completion = [e["completion_tokens"] for e in evaluations if e.get("completion_tokens") is not None]
    total = [e["total_tokens"] for e in evaluations if e.get("total_tokens") is not None]
    costs = [e["estimated_cost_usd"] for e in evaluations if e.get("estimated_cost_usd") is not None]
    usage_rows = sum(1 for e in evaluations if e.get("token_usage_available"))
    return {
        "available": usage_rows > 0,
        "rows_with_usage": usage_rows,
        "prompt_tokens": sum(prompt) if prompt else None,
        "completion_tokens": sum(completion) if completion else None,
        "total_tokens": sum(total) if total else None,
        "estimated_cost_usd": round(sum(costs), 8) if costs else None,
        "status": "ok" if usage_rows else "unavailable",
    }


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.fmean(values), 4) if values else None


def _auto_status(metrics: Dict[str, Any]) -> str:
    retrieval = (metrics or {}).get("retrieval") or {}
    answer = (metrics or {}).get("answer") or {}
    if retrieval.get("status") == "ok" or answer.get("status") == "ok":
        return "ok"
    if retrieval.get("status") == "insufficient_data" or answer.get("status") == "insufficient_data":
        return "insufficient_data"
    return "unavailable"


# --------------------------------------------------------------------------- #
# Automatic evaluation: compute from a request + one system's evaluation
# --------------------------------------------------------------------------- #
def _auto_inputs(
    request: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(question, answer, retrieved_chunks, evidence_chunks)``.

    Baseline RAG answers directly from its retrieved context, so its evidence is
    the retrieved chunks. Self-RAG's evidence is its *final* post-grading evidence
    - kept distinct from the initial retrieval so the two are never conflated.
    """
    question = request.get("question", "")
    answer = evaluation.get("answer") or ""
    system = evaluation.get("system_version")
    if system == BASELINE_SYSTEM:
        chunks = evaluation.get("retrieved_chunks") or request.get("retrieved_chunks") or []
        evidence = chunks
    else:
        chunks = evaluation.get("retrieved_chunks") or []
        evidence = evaluation.get("final_evidence") or []
    return question, answer, chunks, evidence


def build_auto_evaluation(
    request: Dict[str, Any],
    evaluation: Dict[str, Any],
    peers: Optional[List[Dict[str, Any]]] = None,
    path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute (but do not persist) one system's automatic evaluation.

    ``peers`` are the other systems' persisted evaluations for the same request;
    they are required to decide whether Self-RAG's recorded grades can be safely
    reused for Baseline RAG (only when the ranked retrieval is identical).
    """
    question, answer, chunks, evidence = _auto_inputs(request, evaluation)
    metrics = auto_eval.evaluate_system(
        question, answer, chunks, evidence, verification=evaluation
    )
    metrics["version"] = auto_eval.METRICS_VERSION
    metrics["question"] = question
    metrics["system_version"] = evaluation.get("system_version")
    metrics["view"] = SYSTEM_VIEW.get(evaluation.get("system_version"), "retrieval")
    if evaluation.get("system_version") == BASELINE_SYSTEM:
        metrics["grader_ranking"] = _baseline_grader_ranking(request, evaluation, peers, path)
    else:
        metrics["grader_ranking"] = _self_rag_grader_ranking(evaluation)
    return metrics


def _peer_evaluation(evaluations: Optional[List[Dict[str, Any]]], system: str) -> Optional[Dict[str, Any]]:
    for row in evaluations or []:
        if row.get("system_version") == system:
            return row
    return None


def _self_rag_grader_ranking(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Self-RAG's grader-based ranking metrics from its own recorded grades.

    Uses the recorded *initial* ranked retrieval and the grades recorded for those
    chunks. When no ranked retrieval was recorded (e.g. a web-only run), the
    recorded grade order is used as the ranked sequence so web judgments are not
    thrown away.
    """
    chunks = evaluation.get("retrieved_chunks") or []
    grades = evaluation.get("grades") or []
    if not chunks and grades:
        ordered = sorted(
            grades,
            key=lambda g: (
                g.get("retrieval_round") if isinstance(g.get("retrieval_round"), int) else 1,
                g.get("rank") if isinstance(g.get("rank"), int) else 0,
            ),
        )
        ranked = [
            {
                "rank": index + 1,
                "source": grade.get("source"),
                "title": grade.get("title"),
                "type": grade.get("type"),
                "chunk_id": grade.get("chunk_id"),
                "text": "",
            }
            for index, grade in enumerate(ordered)
        ]
        return auto_eval.grader_ranking_metrics(ranked, grades, source="self_rag_grader")
    return auto_eval.grader_ranking_metrics(chunks, grades, source="self_rag_grader")


def _baseline_grader_ranking(
    request: Dict[str, Any],
    evaluation: Dict[str, Any],
    peers: Optional[List[Dict[str, Any]]],
    path: Optional[Any],
) -> Dict[str, Any]:
    """Baseline RAG's grader-based ranking metrics.

    Baseline grades are obtained in exactly two ways, never by copying names:
    1. If Baseline RAG and Self-RAG retrieved the *identical* ranked chunks (same
       ids and text), Self-RAG's recorded grades are reused (verified identity).
    2. Otherwise, grades produced by the optional, explicitly-configured baseline
       chunk grader (stored with :data:`auto_eval.GRADER_METHOD`) are used.
    When neither is available the block is ``not_evaluated`` with a reason.
    """
    chunks = evaluation.get("retrieved_chunks") or request.get("retrieved_chunks") or []
    peer = _peer_evaluation(peers, SELF_RAG_SYSTEM)
    if chunks and peer and auto_eval.same_ranked_retrieval(chunks, peer.get("retrieved_chunks") or []):
        block = auto_eval.grader_ranking_metrics(
            chunks, peer.get("grades") or [], source="baseline_grader_reused"
        )
        block["reuse_note"] = auto_eval.grader_reuse_note("Self-RAG")
        return block

    stored = live_db.get_auto_evaluation(
        evaluation.get("request_id"), BASELINE_SYSTEM, method=auto_eval.GRADER_METHOD, path=path
    )
    stored_grades = ((stored or {}).get("metrics") or {}).get("grades") if stored else None
    if chunks and stored_grades:
        block = auto_eval.grader_ranking_metrics(
            chunks, stored_grades, source="baseline_grader_llm"
        )
        block["reuse_note"] = (
            "Grades produced by the optional, explicitly-configured baseline chunk grader."
        )
        return block

    if not chunks:
        return auto_eval.grader_metrics_not_evaluated(
            "no_retrieved_chunks",
            "Baseline RAG recorded no retrieved chunks for this question, so no "
            "grader-based ranking metrics can be computed.",
        )
    return auto_eval.grader_metrics_not_evaluated(
        "baseline_grader_not_run",
        "Baseline RAG's retrieved chunks have not been graded. Its retrieval differs "
        "from Self-RAG's, so Self-RAG's grades cannot be reused, and the optional "
        "baseline grader was not run.",
    )


def refresh_auto_evaluation(
    request_id: str,
    system_version: str,
    path: Optional[Any] = None,
    peers: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute and persist the automatic evaluation for one request/system.

    Idempotent: the ``UNIQUE(request_id, system_version, method)`` upsert means a
    refresh edits the stored row instead of duplicating it.
    """
    request = live_db.get_request(request_id, path)
    if not request:
        return None
    evaluation = _find_evaluation(request_id, system_version, path)
    if not evaluation:
        return None
    if peers is None:
        peers = live_db.list_evaluations(request_id=request_id, path=path)
    metrics = build_auto_evaluation(request, evaluation, peers, path)
    live_db.save_auto_evaluation(
        {
            "request_id": request_id,
            "system_version": system_version,
            "created_at": evaluation.get("created_at"),
            "method": metrics["method"],
            "status": _auto_status(metrics),
            "metrics": metrics,
        },
        path,
    )
    return metrics


def _find_evaluation(
    request_id: str,
    system_version: str,
    path: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    for row in live_db.list_evaluations(request_id=request_id, path=path):
        if row.get("system_version") == system_version:
            return row
    return None


def _stored_or_compute(
    request: Dict[str, Any],
    evaluation: Dict[str, Any],
    peers: Optional[List[Dict[str, Any]]] = None,
    path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reuse a stored automatic evaluation, else compute and persist it once.

    A stored row written by an older metrics version is recomputed (upserted under
    the same key, so it is never duplicated).
    """
    stored = live_db.get_auto_evaluation(
        request["request_id"], evaluation["system_version"], method=auto_eval.METHOD, path=path
    )
    if stored and stored.get("metrics") and stored["metrics"].get("version") == auto_eval.METRICS_VERSION:
        return stored["metrics"]
    metrics = refresh_auto_evaluation(
        request["request_id"], evaluation["system_version"], path, peers
    )
    return metrics or build_auto_evaluation(request, evaluation, peers, path)


# --------------------------------------------------------------------------- #
# Row shaping (only what the aggregate builders still need)
# --------------------------------------------------------------------------- #
def _shape_rows(evaluations: List[Dict[str, Any]], requests_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for e in evaluations:
        request = requests_by_id.get(e["request_id"], {})
        rows.append(
            {
                "request_id": e["request_id"],
                "system_version": e["system_version"],
                "question": request.get("question", ""),
                "created_at": e["created_at"],
                "request_failed": e.get("status") != "ok",
                "error_category": e.get("error_category") or "",
                "error": e.get("error") or "",
                "latency_seconds": e.get("latency_seconds"),
                "prompt_tokens": e.get("prompt_tokens"),
                "completion_tokens": e.get("completion_tokens"),
                "total_tokens": e.get("total_tokens"),
                "token_usage_available": e.get("token_usage_available"),
                "estimated_cost_usd": e.get("estimated_cost_usd"),
                "route": e.get("route") or "",
                "support_status": e.get("support_status") or "",
                "usefulness": e.get("usefulness") or "",
                "used_web_search": e.get("used_web_search"),
                "retries": e.get("retries"),
                "answer": e.get("answer") or "",
            }
        )
    return rows


def _system_block(
    rows: List[Dict[str, Any]],
    system: str,
    path: Optional[Any] = None,
) -> Dict[str, Any]:
    system_rows = [r for r in rows if r["system_version"] == system]
    if not system_rows:
        return {
            "system_version": system,
            "evaluated": 0,
            "success_rate": None,
            "failure_rate": None,
            "failure_categories": {},
            "latency_seconds": _latency([]),
            "tokens": _tokens([]),
            "retries": {"average": None, "rate": None, "sample": 0, "status": "insufficient_data"},
            "web_search_rate": None,
            "verification_rate": None,
            "automatic": _system_automatic(system, path),
            "status": "insufficient_data",
        }
    auto = em.aggregate_system(system_rows)["automatic"]
    lat = auto["latency_seconds"]["successful"]
    retries = [r["retries"] for r in system_rows if r.get("retries") is not None]
    web_rows = [r for r in system_rows if r.get("used_web_search") is not None]
    verified = [r for r in system_rows if (r.get("support_status") or "")]
    return {
        "system_version": system,
        "evaluated": len(system_rows),
        "success_rate": auto["success_rate"],
        "failure_rate": auto["failure_rate"],
        "failure_categories": auto["failure_categories"],
        "latency_seconds": {
            "average_seconds": lat["mean"],
            "p50_seconds": lat["p50"],
            "p95_seconds": lat["p95"],
            "min_seconds": lat["min"],
            "max_seconds": lat["max"],
            "sample": lat["n"],
            "status": "ok" if lat["n"] else "insufficient_data",
        },
        "tokens": _tokens(system_rows),
        "retries": {
            "average": round(statistics.fmean(retries), 4) if retries else None,
            "rate": _rate(sum(1 for r in retries if r > 0), len(system_rows)),
            "sample": len(retries),
            "status": "ok" if retries else "insufficient_data",
        },
        "web_search_rate": _rate(sum(1 for r in web_rows if r.get("used_web_search")), len(web_rows)),
        "verification_rate": _rate(len(verified), len(system_rows)),
        "automatic": _system_automatic(system, path),
        "status": "ok",
    }


def _system_automatic(system: str, path: Optional[Any] = None) -> Dict[str, Any]:
    """Aggregate the stored automatic evaluations for one system (proxies only)."""
    coverages, answer_rel, support = [], [], []
    evaluated = 0
    for row in live_db.list_auto_evaluations(path=path):
        if row.get("system_version") != system or row.get("method") != auto_eval.METHOD:
            continue
        evaluated += 1
        metrics = row.get("metrics") or {}
        retrieval = metrics.get("retrieval") or {}
        answer = metrics.get("answer") or {}
        if isinstance(retrieval.get("mean_query_term_coverage"), (int, float)):
            coverages.append(retrieval["mean_query_term_coverage"])
        if isinstance(answer.get("answer_relevance"), (int, float)):
            answer_rel.append(answer["answer_relevance"])
        if isinstance(answer.get("evidence_support"), (int, float)):
            support.append(answer["evidence_support"])
    return {
        "method": auto_eval.METHOD,
        "evaluated": evaluated,
        "mean_query_term_coverage": _mean(coverages),
        "mean_answer_relevance": _mean(answer_rel),
        "mean_evidence_support": _mean(support),
        "samples": {
            "retrieval": len(coverages),
            "answer_relevance": len(answer_rel),
            "evidence_support": len(support),
        },
        "status": "ok" if evaluated else "insufficient_data",
        "traditional_metrics": auto_eval.traditional_metrics_unavailable(),
        "note": AUTOMATIC_NOTE,
    }


def _self_rag_workflow(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    self_rows = [r for r in rows if r["system_version"] == SELF_RAG_SYSTEM]
    if not self_rows:
        return {"evaluated": 0, "status": "insufficient_data"}
    support_counts = {"fully_supported": 0, "partially_supported": 0, "no_support": 0}
    for r in self_rows:
        status = r.get("support_status") or ""
        if status in support_counts:
            support_counts[status] += 1
    usefulness_counts = {"useful": 0, "not_useful": 0}
    for r in self_rows:
        status = r.get("usefulness") or ""
        if status in usefulness_counts:
            usefulness_counts[status] += 1
    web_rows = [r for r in self_rows if r.get("used_web_search") is not None]
    retries = [r["retries"] for r in self_rows if r.get("retries") is not None]
    total = len(self_rows)
    verified = sum(1 for r in self_rows if (r.get("support_status") or ""))
    return {
        "evaluated": total,
        "retry_rate": _rate(sum(1 for r in retries if r > 0), total),
        "web_search_rate": _rate(sum(1 for r in web_rows if r.get("used_web_search")), total),
        "verification_rate": _rate(verified, total),
        "support_status": {**support_counts, "counts_total": sum(support_counts.values())},
        "usefulness": {**usefulness_counts, "counts_total": sum(usefulness_counts.values())},
        "status": "ok",
        "note": "Reported Self-RAG verification decisions, not ground-truth correctness.",
    }


def _quality(evaluations: List[Dict[str, Any]], quality_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    correctness = [q["answer_correctness"] for q in quality_scores if q.get("answer_correctness") is not None]
    faithfulness = [q["faithfulness"] for q in quality_scores if q.get("faithfulness") is not None]
    methods = sorted({str(q.get("method") or "unknown") for q in quality_scores})

    if correctness or faithfulness:
        status = "ok"
        reason = None
    elif evaluations:
        status = "pending"
        reason = "evaluator_not_run"
    else:
        status = "insufficient_data"
        reason = "no_evaluated_requests"

    return {
        "answer_correctness": {
            "mean": _mean(correctness),
            "n": len(correctness),
            "status": "ok" if correctness else status,
        },
        "faithfulness": {
            "mean": _mean(faithfulness),
            "n": len(faithfulness),
            "status": "ok" if faithfulness else status,
        },
        "methods": methods,
        "status": status,
        "reason": reason,
        "note": (
            "Answer correctness / faithfulness appear only when the optional, "
            "explicitly-configured evaluator has stored scores. Absent scores are "
            "reported as unavailable - never as zero."
        ),
    }


# --------------------------------------------------------------------------- #
# Public aggregate builders (retained for the API; not shown on the dashboard)
# --------------------------------------------------------------------------- #
def build_summary(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    path: Optional[Any] = None,
) -> Dict[str, Any]:
    requests = live_db.list_requests(date_from, date_to, path=path)
    evaluations = live_db.list_evaluations(date_from, date_to, path=path)
    quality_scores = live_db.list_quality_scores(date_from, date_to, path=path)
    requests_by_id = {r["request_id"]: r for r in requests}
    rows = _shape_rows(evaluations, requests_by_id)

    success_latencies = [
        r["latency_seconds"] for r in requests
        if r.get("status") == "ok" and r.get("latency_seconds") is not None
    ]
    evaluated = len({e["request_id"] for e in evaluations})

    counts = _counts(requests)
    return {
        "date_from": date_from,
        "date_to": date_to,
        "requests": {**counts, "evaluated": evaluated},
        "latency": _latency(success_latencies),
        "self_rag": _self_rag_workflow(rows),
        "tokens": _tokens(evaluations),
        "quality": {
            "retrieval": _automatic_aggregate(path),
            "answer": _quality(evaluations, quality_scores),
        },
        "legend": LEGEND,
    }


def _automatic_aggregate(path: Optional[Any] = None) -> Dict[str, Any]:
    coverages, answer_rel, support = [], [], []
    evaluated = 0
    for row in live_db.list_auto_evaluations(path=path):
        if row.get("method") != auto_eval.METHOD:
            continue
        evaluated += 1
        metrics = row.get("metrics") or {}
        retrieval = metrics.get("retrieval") or {}
        answer = metrics.get("answer") or {}
        if isinstance(retrieval.get("mean_query_term_coverage"), (int, float)):
            coverages.append(retrieval["mean_query_term_coverage"])
        if isinstance(answer.get("answer_relevance"), (int, float)):
            answer_rel.append(answer["answer_relevance"])
        if isinstance(answer.get("evidence_support"), (int, float)):
            support.append(answer["evidence_support"])
    return {
        "method": auto_eval.METHOD,
        "evaluated_systems": evaluated,
        "mean_query_term_coverage": _mean(coverages),
        "mean_answer_relevance": _mean(answer_rel),
        "mean_evidence_support": _mean(support),
        "samples": {
            "retrieval": len(coverages),
            "answer_relevance": len(answer_rel),
            "evidence_support": len(support),
        },
        "status": "ok" if evaluated else "insufficient_data",
        "traditional_metrics": auto_eval.traditional_metrics_unavailable(),
        "note": AUTOMATIC_NOTE,
    }


def build_comparison(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    path: Optional[Any] = None,
) -> Dict[str, Any]:
    requests = live_db.list_requests(date_from, date_to, path=path)
    evaluations = live_db.list_evaluations(date_from, date_to, path=path)
    requests_by_id = {r["request_id"]: r for r in requests}
    rows = _shape_rows(evaluations, requests_by_id)
    return {
        "date_from": date_from,
        "date_to": date_to,
        "systems": {
            BASELINE_SYSTEM: _system_block(rows, BASELINE_SYSTEM, path),
            SELF_RAG_SYSTEM: _system_block(rows, SELF_RAG_SYSTEM, path),
        },
        "note": (
            "Both systems get the same automatic evaluation rules. Baseline RAG is "
            "assessed on its retrieved context; Self-RAG is assessed on its final "
            "evidence. Conventional retrieval metrics are unavailable without a "
            "relevance reference."
        ),
    }


def build_daily(date_from: Optional[str] = None, date_to: Optional[str] = None, path: Optional[Any] = None) -> Dict[str, Any]:
    requests = live_db.list_requests(date_from, date_to, path=path)
    evaluations = live_db.list_evaluations(date_from, date_to, path=path)
    evals_by_day: Dict[str, int] = {}
    for e in evaluations:
        day = (e.get("created_at") or "")[:10]
        evals_by_day[day] = evals_by_day.get(day, 0) + 1

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in requests:
        day = (r.get("created_at") or "")[:10]
        buckets.setdefault(day, []).append(r)

    points = []
    for day in sorted(buckets):
        day_requests = buckets[day]
        counts = _counts(day_requests)
        latencies = [
            r["latency_seconds"] for r in day_requests
            if r.get("status") == "ok" and r.get("latency_seconds") is not None
        ]
        points.append(
            {
                "date": day,
                "requests": counts["total"],
                "success": counts["success"],
                "failed": counts["failed"],
                "success_rate": counts["success_rate"],
                "failure_rate": counts["failure_rate"],
                "evaluations": evals_by_day.get(day, 0),
                "latency": _latency(latencies),
                "status": "ok",
            }
        )
    return {
        "date_from": date_from,
        "date_to": date_to,
        "points": points,
        "status": "ok" if points else "insufficient_data",
    }


# --------------------------------------------------------------------------- #
# Question history (the only thing the dashboard shows)
# --------------------------------------------------------------------------- #
def build_history(
    page: int = 1,
    page_size: int = 20,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    system: Optional[str] = None,
    path: Optional[Any] = None,
) -> Dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 100))
    offset = (page - 1) * page_size

    total = live_db.count_requests(date_from, date_to, status=status, system=system, path=path)
    requests = live_db.list_requests(
        date_from, date_to, status=status, system=system, limit=page_size, offset=offset, path=path
    )
    evaluations = live_db.list_evaluations(date_from, date_to, path=path)
    auto_by_key = {
        (row["request_id"], row["system_version"]): row
        for row in live_db.list_auto_evaluations(path=path)
        if row.get("method") == auto_eval.METHOD
    }

    by_request: Dict[str, List[Dict[str, Any]]] = {}
    for e in evaluations:
        by_request.setdefault(e["request_id"], []).append(e)

    items = []
    for r in requests:
        evals = by_request.get(r["request_id"], [])
        by_system = {e["system_version"]: e for e in evals}
        items.append(
            {
                "request_id": r["request_id"],
                "created_at": r["created_at"],
                "question": r["question"],
                "status": r["status"],
                "latency_seconds": r.get("latency_seconds"),
                "error": r.get("error"),
                "error_category": r.get("error_category"),
                "self_rag": _history_system(by_system.get(SELF_RAG_SYSTEM), auto_by_key),
                "baseline": _history_system(by_system.get(BASELINE_SYSTEM), auto_by_key),
                "evaluation": _question_evaluation(evals),
            }
        )

    total_pages = (total + page_size - 1) // page_size if total else 0
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "items": items,
        "status": "ok" if items else "insufficient_data",
    }


def _question_evaluation(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A concise, per-question evaluation status for one history row."""
    systems = {e["system_version"] for e in evaluations}
    has_baseline = BASELINE_SYSTEM in systems
    has_self_rag = SELF_RAG_SYSTEM in systems
    if has_baseline and has_self_rag:
        label, status = "Both systems evaluated", "ok"
    elif has_self_rag:
        label, status = "Self-RAG evaluated", "partial"
    elif has_baseline:
        label, status = "Baseline evaluated", "partial"
    else:
        label, status = "Not evaluated", "insufficient_data"
    return {"label": label, "status": status}


def _history_system(
    evaluation: Optional[Dict[str, Any]],
    auto_by_key: Dict[Tuple[str, str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not evaluation:
        return None
    auto = auto_by_key.get((evaluation["request_id"], evaluation["system_version"]))
    return {
        "system_version": evaluation["system_version"],
        "status": evaluation["status"],
        "latency_seconds": evaluation.get("latency_seconds"),
        "route": evaluation.get("route") or "",
        "used_web_search": bool(evaluation.get("used_web_search")),
        "retries": evaluation.get("retries"),
        "error": evaluation.get("error"),
        "error_category": evaluation.get("error_category"),
        "automatic_status": (auto or {}).get("status"),
    }


# --------------------------------------------------------------------------- #
# Per-question detail
# --------------------------------------------------------------------------- #
def _public_chunks(chunks: Any) -> List[Dict[str, Any]]:
    return auto_eval.normalize_chunks(chunks)


def build_request_detail(
    request_id: str,
    path: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    request = live_db.get_request(request_id, path)
    if not request:
        return None
    evaluations = live_db.list_evaluations(request_id=request_id, path=path)
    by_system = {e["system_version"]: e for e in evaluations}
    quality_scores = live_db.list_quality_scores(path=path)
    request_quality = [q for q in quality_scores if q["request_id"] == request_id]

    systems: Dict[str, Any] = {}
    for system, evaluation in by_system.items():
        metrics = _stored_or_compute(request, evaluation, peers=evaluations, path=path)
        if system == BASELINE_SYSTEM:
            chunks = evaluation.get("retrieved_chunks") or request.get("retrieved_chunks") or []
            # Baseline RAG answers directly from its retrieved context, so its
            # evidence *is* its retrieved chunks. It is exposed once (as
            # ``retrieved_chunks``) rather than duplicated as final evidence.
            evidence: List[Dict[str, Any]] = []
        else:
            chunks = evaluation.get("retrieved_chunks") or []
            evidence = evaluation.get("final_evidence") or []
        systems[system] = {
            "system_version": evaluation["system_version"],
            "view": SYSTEM_VIEW.get(system, "retrieval"),
            "status": evaluation["status"],
            "answer": evaluation.get("answer"),
            "route": evaluation.get("route"),
            "latency_seconds": evaluation.get("latency_seconds"),
            "retries": evaluation.get("retries"),
            "support_status": evaluation.get("support_status"),
            "usefulness": evaluation.get("usefulness"),
            "used_web_search": bool(evaluation.get("used_web_search")),
            "used_sources": evaluation.get("used_sources") or [],
            "retrieved_chunks": _public_chunks(chunks),
            "final_evidence": _public_chunks(evidence),
            "grades": evaluation.get("grades") or [],
            "verification": metrics.get("verification") or {},
            "automatic": metrics.get("retrieval") or {},
            "grader_ranking": metrics.get("grader_ranking")
            or auto_eval.grader_metrics_not_evaluated(
                "not_computed", "No grader-based ranking metrics are available for this system."
            ),
            "answer_quality": metrics.get("answer") or {},
            "traditional_metrics": metrics.get("traditional_metrics")
            or auto_eval.traditional_metrics_unavailable(),
            "error": evaluation.get("error"),
            "error_category": evaluation.get("error_category"),
            "tokens": {
                "prompt_tokens": evaluation.get("prompt_tokens"),
                "completion_tokens": evaluation.get("completion_tokens"),
                "total_tokens": evaluation.get("total_tokens"),
                "available": bool(evaluation.get("token_usage_available")),
                "estimated_cost_usd": evaluation.get("estimated_cost_usd"),
            },
        }

    return {
        "request_id": request["request_id"],
        "created_at": request["created_at"],
        "thread_id": request.get("thread_id"),
        "question": request["question"],
        "status": request["status"],
        "latency_seconds": request.get("latency_seconds"),
        "error": request.get("error"),
        "error_category": request.get("error_category"),
        "answer": request.get("answer"),
        "route": request.get("route"),
        "used_web_search": bool(request.get("used_web_search")),
        "support_status": request.get("support_status"),
        "usefulness": request.get("usefulness"),
        "retries": request.get("retries"),
        "retrieved_chunks": _public_chunks(request.get("retrieved_chunks")),
        "shared_initial_retrieval": bool(
            systems.get(BASELINE_SYSTEM)
            and systems.get(SELF_RAG_SYSTEM)
            and auto_eval.same_ranked_retrieval(
                systems[BASELINE_SYSTEM]["retrieved_chunks"],
                systems[SELF_RAG_SYSTEM]["retrieved_chunks"],
            )
        ),
        "retrieved_sources": request.get("retrieved_sources") or [],
        "retrieval_available": bool(request.get("retrieval_available")),
        "sources": request.get("sources") or [],
        "trace": request.get("trace") or [],
        "tokens": {
            "prompt_tokens": request.get("prompt_tokens"),
            "completion_tokens": request.get("completion_tokens"),
            "total_tokens": request.get("total_tokens"),
            "available": bool(request.get("token_usage_available")),
            "estimated_cost_usd": request.get("estimated_cost_usd"),
        },
        "systems": systems,
        "method": auto_eval.METHOD,
        "metrics_version": auto_eval.METRICS_VERSION,
        "method_note": AUTOMATIC_NOTE,
        "grader_method": auto_eval.GRADER_METHOD,
        "grader_methodology": auto_eval.GRADER_FORMULAS,
        "grader_limitations": auto_eval.GRADER_LIMITATIONS,
        "traditional_metrics": auto_eval.traditional_metrics_unavailable(),
        "quality_scores": request_quality,
        "quality_status": ("ok" if request_quality else ("pending" if evaluations else "insufficient_data")),
    }


# --------------------------------------------------------------------------- #
# Optional, explicitly-configured evaluator (bounded; one stored score per pair)
# --------------------------------------------------------------------------- #
def judge_request(
    request_id: str,
    system_version: str,
    path: Optional[Any] = None,
    settings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the optional LLM judge once for one request/system. Never fabricates.

    Returns a dict whose ``status`` is one of ``ok``, ``already_scored``,
    ``not_configured``, ``not_found``, ``no_evaluation`` or ``failed``.
    """
    request = live_db.get_request(request_id, path)
    if not request:
        return {"status": "not_found"}
    evaluation = _find_evaluation(request_id, system_version, path)
    if not evaluation:
        return {"status": "no_evaluation"}

    existing = live_db.get_auto_evaluation(
        request_id, system_version, method=auto_eval.JUDGE_METHOD, path=path
    )
    if existing and existing.get("metrics"):
        return {"status": "already_scored", "score": existing["metrics"]}

    if not auto_eval.judge_enabled(settings):
        return {"status": "not_configured"}

    question, answer, chunks, evidence = _auto_inputs(request, evaluation)
    evidence = evidence or chunks
    try:
        score = auto_eval.judge_answer(question, answer, evidence, settings=settings)
    except Exception as exc:  # noqa: BLE001 - a judge failure is not an answer failure
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    live_db.save_auto_evaluation(
        {
            "request_id": request_id,
            "system_version": system_version,
            "created_at": evaluation.get("created_at"),
            "method": auto_eval.JUDGE_METHOD,
            "status": "ok",
            "metrics": score,
        },
        path,
    )
    return {"status": "ok", "score": score}


def grade_baseline_request(
    request_id: str,
    path: Optional[Any] = None,
    settings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Grade Baseline RAG's retrieved chunks once with the optional evaluator.

    Bounded to one stored grading per request (method
    :data:`auto_eval.GRADER_METHOD`) and only runs when the evaluator is
    explicitly configured. Never fabricates: a missing configuration or a grader
    failure is reported as a status, not a score. When the grades are stored, the
    baseline automatic evaluation is recomputed so the dashboard can show its
    grader-based ranking metrics.
    """
    request = live_db.get_request(request_id, path)
    if not request:
        return {"status": "not_found"}
    evaluation = _find_evaluation(request_id, BASELINE_SYSTEM, path)
    if not evaluation:
        return {"status": "no_evaluation"}

    existing = live_db.get_auto_evaluation(
        request_id, BASELINE_SYSTEM, method=auto_eval.GRADER_METHOD, path=path
    )
    if existing and existing.get("metrics"):
        return {"status": "already_graded", "grades": existing["metrics"].get("grades", [])}

    if not auto_eval.judge_enabled(settings):
        return {"status": "not_configured"}

    chunks = evaluation.get("retrieved_chunks") or request.get("retrieved_chunks") or []
    if not chunks:
        return {"status": "no_chunks"}

    try:
        grades = auto_eval.grade_chunks(request.get("question", ""), chunks, settings=settings)
    except Exception as exc:  # noqa: BLE001 - a grader failure is not an answer failure
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    live_db.save_auto_evaluation(
        {
            "request_id": request_id,
            "system_version": BASELINE_SYSTEM,
            "created_at": evaluation.get("created_at"),
            "method": auto_eval.GRADER_METHOD,
            "status": "ok",
            "metrics": {"grades": grades},
        },
        path,
    )
    # Recompute the baseline automatic evaluation so its grader block includes the
    # freshly stored grades (upserted under the same key, never duplicated).
    refresh_auto_evaluation(request_id, BASELINE_SYSTEM, path)
    return {"status": "ok", "grades": grades}
