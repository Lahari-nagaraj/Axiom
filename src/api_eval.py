"""FastAPI router for the live evaluation dashboard.

All endpoints read persisted runtime records via ``src.live_metrics``. None of
them trigger answer generation or touch the benchmark dataset. The automatic
evaluation is deterministic and local; the optional ``/judge`` endpoint only runs
an LLM evaluator when it has been explicitly configured. Response schemas live in
``src.models``. Invalid dates return ``400`` and unknown request ids ``404``.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src import live_metrics
from src.auto_eval import judge_enabled
from src.models import (
    DailyMetricsResponse,
    EvaluationComparisonResponse,
    EvaluationHistoryResponse,
    EvaluationSummaryResponse,
    GradeBaselineResponse,
    JudgeResponse,
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


def _validate_date(value: Optional[str], name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        date_cls.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {name} '{value}'; expected an ISO date (YYYY-MM-DD).",
        )
    return value


@router.get("/summary", response_model=EvaluationSummaryResponse)
def evaluation_summary(
    date_from: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
    date_to: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
):
    return live_metrics.build_summary(
        _validate_date(date_from, "date_from"), _validate_date(date_to, "date_to")
    )


@router.get("/comparison", response_model=EvaluationComparisonResponse)
def evaluation_comparison(
    date_from: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
    date_to: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
):
    return live_metrics.build_comparison(
        _validate_date(date_from, "date_from"), _validate_date(date_to, "date_to")
    )


@router.get("/daily", response_model=DailyMetricsResponse)
def evaluation_daily(
    date_from: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
    date_to: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
):
    return live_metrics.build_daily(
        _validate_date(date_from, "date_from"), _validate_date(date_to, "date_to")
    )


@router.get("/history", response_model=EvaluationHistoryResponse)
def evaluation_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    date_from: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
    date_to: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD), inclusive."),
    status: Optional[str] = Query(None, pattern="^(ok|failed)$"),
    system: Optional[str] = Query(None, description="Filter to requests evaluated by this system."),
):
    return live_metrics.build_history(
        page=page,
        page_size=page_size,
        date_from=_validate_date(date_from, "date_from"),
        date_to=_validate_date(date_to, "date_to"),
        status=status,
        system=system,
    )


@router.get("/requests/{request_id}")
def evaluation_request_detail(request_id: str):
    detail = live_metrics.build_request_detail(request_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"request '{request_id}' not found")
    return detail


@router.post("/requests/{request_id}/judge", response_model=JudgeResponse)
def evaluation_request_judge(
    request_id: str,
    system: str = Query(..., description="System version to judge, e.g. baseline_rag or axiom_self_rag."),
):
    """Run the optional, explicitly-configured LLM evaluator once for one system.

    Bounded to a single stored score per ``(request, system)``. Returns ``503``
    when the evaluator is not configured and never fabricates a score.
    """
    result = live_metrics.judge_request(request_id, system)
    status = result.get("status")
    if status == "not_found":
        raise HTTPException(status_code=404, detail=f"request '{request_id}' not found")
    if status == "no_evaluation":
        raise HTTPException(status_code=404, detail=f"no '{system}' evaluation for request '{request_id}'")
    if status == "not_configured":
        raise HTTPException(
            status_code=503,
            detail=(
                "The optional LLM evaluator is not configured. Set live_eval_judge_enabled "
                "and a provider key to use it; the deterministic automatic evaluation "
                "remains available without it."
            ),
        )
    if status == "failed":
        raise HTTPException(status_code=502, detail=f"evaluator failed: {result.get('error')}")
    return {
        "request_id": request_id,
        "system_version": system,
        "status": status,
        "score": result.get("score"),
    }


@router.get("/judge/status")
def evaluation_judge_status():
    """Whether the optional LLM evaluator is configured (no model call)."""
    return {"enabled": judge_enabled()}


@router.post("/requests/{request_id}/grade-baseline", response_model=GradeBaselineResponse)
def evaluation_request_grade_baseline(request_id: str):
    """Grade Baseline RAG's retrieved chunks once with the optional evaluator.

    Bounded to a single stored grading per request; returns ``503`` when the
    evaluator is not configured and never fabricates grades.
    """
    result = live_metrics.grade_baseline_request(request_id)
    status = result.get("status")
    if status == "not_found":
        raise HTTPException(status_code=404, detail=f"request '{request_id}' not found")
    if status == "no_evaluation":
        raise HTTPException(
            status_code=404, detail=f"no baseline_rag evaluation for request '{request_id}'"
        )
    if status == "no_chunks":
        raise HTTPException(
            status_code=409, detail="Baseline RAG recorded no retrieved chunks to grade."
        )
    if status == "not_configured":
        raise HTTPException(
            status_code=503,
            detail=(
                "The optional baseline chunk grader is not configured. Set "
                "live_eval_judge_enabled and a provider key to use it; Self-RAG's "
                "recorded grades are reused automatically when the retrieval is identical."
            ),
        )
    if status == "failed":
        raise HTTPException(status_code=502, detail=f"grader failed: {result.get('error')}")
    return {
        "request_id": request_id,
        "system_version": "baseline_rag",
        "status": status,
        "grades": result.get("grades", []),
    }
