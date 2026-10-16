"""Runtime request logging and Baseline RAG shadow evaluation.

Flow for every real ``POST /api/chat`` request:

1. ``record_chat`` persists the request record and the Self-RAG evaluation
   *synchronously* (wrapped so a logging failure can never break the chat).
2. ``schedule_baseline_shadow`` queues the Baseline RAG evaluation to run
   *after* the response is sent, via FastAPI ``BackgroundTasks``.

Baseline execution
------------------
Durable background workers (Celery/RQ) are not part of this deployment, so the
supported alternative is FastAPI's ``BackgroundTasks``: the task runs in the
process after the response has been returned to the user. A module-level
semaphore bounds how many baseline evaluations run concurrently so a burst of
requests cannot spawn uncontrolled provider calls. Because this is in-process,
queued tasks do not survive a process restart; see ``README`` limitations.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import get_settings
from src import live_db

logger = logging.getLogger("axiom.live_eval")

SELF_RAG_SYSTEM = "axiom_self_rag"
BASELINE_SYSTEM = "baseline_rag"

_BASELINE = None
_BASELINE_LOCK = threading.Lock()
_SEMAPHORE: Optional[threading.Semaphore] = None
_SEMAPHORE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def new_request_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_retriever():
    from src.vectorstore import get_retriever

    return get_retriever()


def _get_baseline():
    global _BASELINE
    if _BASELINE is None:
        with _BASELINE_LOCK:
            if _BASELINE is None:
                from evaluation.run_evaluation import BaselineRAG

                _BASELINE = BaselineRAG()
    return _BASELINE


def _semaphore() -> threading.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        with _SEMAPHORE_LOCK:
            if _SEMAPHORE is None:
                limit = max(1, int(getattr(get_settings(), "live_eval_max_concurrency", 2) or 1))
                _SEMAPHORE = threading.Semaphore(limit)
    return _SEMAPHORE


def _sanitize_error(exc: BaseException) -> str:
    try:
        from evaluation.run_evaluation import sanitize, _secret_values

        return sanitize(f"{type(exc).__name__}: {exc}", _secret_values())
    except Exception:  # noqa: BLE001 - sanitising must never fail the caller
        return f"{type(exc).__name__}: {exc}"


def _classify_error(exc: BaseException) -> str:
    try:
        from evaluation.run_evaluation import classify_error

        return classify_error(exc)
    except Exception:  # noqa: BLE001
        return "unknown"


def _count_retries(trace: List[str]) -> int:
    try:
        from evaluation.run_evaluation import count_axiom_retries

        return count_axiom_retries(trace or [])
    except Exception:  # noqa: BLE001
        return 0


def _source_names(docs: List[Any]) -> List[str]:
    try:
        from evaluation.run_evaluation import retrieved_source_names

        return retrieved_source_names(docs)
    except Exception:  # noqa: BLE001
        return []


def _result_source_names(sources: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for item in sources or []:
        if not isinstance(item, dict):
            continue
        name = item.get("title") or item.get("source") or ""
        if name:
            names.append(str(name))
    return names


# Execution artifacts persisted for the automatic evaluation. Both bounds keep a
# single request from bloating the database.
_CHUNK_TEXT_LIMIT = 2000
_MAX_CHUNKS = 8


def _chunk_artifacts(chunks: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Sanitize already-recorded chunk dicts (from ``run_self_rag``) before storage."""
    out: List[Dict[str, Any]] = []
    for index, raw in enumerate(chunks or []):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "")[:_CHUNK_TEXT_LIMIT]
        chunk_id = str(raw.get("chunk_id") or "")
        out.append(
            {
                "rank": raw.get("rank", index + 1),
                "source": str(raw.get("source") or ""),
                "title": str(raw.get("title") or ""),
                "type": str(raw.get("type") or "internal"),
                "page": raw.get("page"),
                "text": text,
                "chunk_id": chunk_id,
            }
        )
        if len(out) >= _MAX_CHUNKS:
            break
    return out


def _grade_artifacts(grades: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in grades or []:
        if not isinstance(raw, dict):
            continue
        out.append(
            {
                "chunk_id": str(raw.get("chunk_id") or ""),
                "source": str(raw.get("source") or ""),
                "title": str(raw.get("title") or ""),
                "type": str(raw.get("type") or "internal"),
                "relevant": bool(raw.get("relevant")),
                "rank": raw.get("rank"),
                "retrieval_round": raw.get("retrieval_round"),
            }
        )
    return out


def _docs_to_chunks(docs: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Convert retrieved ``Document`` objects into persistable chunk records."""
    from src.auto_eval import chunk_identity

    out: List[Dict[str, Any]] = []
    for index, doc in enumerate(docs or []):
        meta = getattr(doc, "metadata", {}) or {}
        typ = "web" if meta.get("source_type") == "web" else "internal"
        page = meta.get("page")
        try:
            page = int(page) + 1 if typ == "internal" and page is not None else None
        except (TypeError, ValueError):
            page = None
        source = str(meta.get("document_name") or meta.get("source") or meta.get("url") or "")
        text = str(getattr(doc, "page_content", "") or "")[:_CHUNK_TEXT_LIMIT]
        out.append(
            {
                "rank": index + 1,
                "source": source,
                "title": str(meta.get("title") or meta.get("document_name") or ""),
                "type": typ,
                "page": page,
                "text": text,
                "chunk_id": chunk_identity(typ, source, text),
            }
        )
        if len(out) >= _MAX_CHUNKS:
            break
    return out


def _refresh_auto_evaluation(request_id: str, system_version: str, path: Optional[Any] = None) -> None:
    """Compute and persist the automatic evaluation. Never raises."""
    try:
        from src import live_metrics

        live_metrics.refresh_auto_evaluation(request_id, system_version, path)
    except Exception:  # noqa: BLE001 - evaluation must never break the request
        logger.exception("automatic evaluation failed for request_id=%s", request_id)


def estimate_cost(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    """Estimate USD cost from provider tokens and configured pricing.

    Returns ``None`` (unavailable) unless pricing is explicitly configured via
    ``GROQ_INPUT_COST_PER_1M`` / ``GROQ_OUTPUT_COST_PER_1M`` and usage exists.
    Never invents a number.
    """
    if prompt_tokens is None and completion_tokens is None:
        return None
    settings = get_settings()
    in_price = getattr(settings, "groq_input_cost_per_1m", None)
    out_price = getattr(settings, "groq_output_cost_per_1m", None)
    if in_price is None and out_price is None:
        return None
    total = 0.0
    counted = False
    if prompt_tokens is not None and in_price is not None:
        total += prompt_tokens / 1_000_000 * in_price
        counted = True
    if completion_tokens is not None and out_price is not None:
        total += completion_tokens / 1_000_000 * out_price
        counted = True
    return round(total, 8) if counted else None


def _usage_from_result(result: Dict[str, Any]):
    """Extract (prompt, completion, total, available) from a Self-RAG result.

    ``run_self_rag`` does not currently expose provider usage, so this is
    normally unavailable; the hook exists so usage is recorded if it is added.
    """
    usage = result.get("usage") if isinstance(result, dict) else None
    if isinstance(usage, dict) and usage:
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
    else:
        prompt = result.get("prompt_tokens")
        completion = result.get("completion_tokens")
        total = result.get("total_tokens")
    available = any(v is not None for v in (prompt, completion, total))
    return prompt, completion, total, available


# --------------------------------------------------------------------------- #
# Request + Self-RAG evaluation logging
# --------------------------------------------------------------------------- #
def _request_record(
    request_id: str,
    created_at: str,
    question: str,
    thread_id: str,
    *,
    status: str,
    latency_seconds: Optional[float],
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    error_category: Optional[str] = None,
) -> Dict[str, Any]:
    result = result or {}
    prompt, completion, total, usage_available = _usage_from_result(result)
    return {
        "request_id": request_id,
        "created_at": created_at,
        "thread_id": thread_id,
        "question": question,
        "system_version": SELF_RAG_SYSTEM,
        "status": status,
        "error": error,
        "error_category": error_category,
        "latency_seconds": latency_seconds,
        "answer": result.get("answer", "") or "",
        "route": result.get("route", "") or "",
        "used_web_search": int(bool(result.get("used_web_search"))),
        "support_status": result.get("support_status", "") or "",
        "usefulness": result.get("usefulness", "") or "",
        "retries": _count_retries(result.get("trace", [])) if result else 0,
        "retrieval_available": 0,
        "retrieved_sources_json": None,
        "sources_json": _json(result.get("sources", [])),
        "trace_json": _json(result.get("trace", [])),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "token_usage_available": int(bool(usage_available)),
        "estimated_cost_usd": estimate_cost(prompt, completion),
    }


def _evaluation_record(
    request_id: str,
    system_version: str,
    created_at: str,
    *,
    status: str,
    latency_seconds: Optional[float],
    answer: str = "",
    error: Optional[str] = None,
    error_category: Optional[str] = None,
    route: str = "",
    retries: Optional[int] = None,
    support_status: str = "",
    usefulness: str = "",
    used_web_search: bool = False,
    retrieval_available: bool = False,
    retrieved_sources: Optional[List[str]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    used_sources: Optional[List[str]] = None,
    final_evidence: Optional[List[Dict[str, Any]]] = None,
    grades: Optional[List[Dict[str, Any]]] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    token_usage_available: bool = False,
    estimated_cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "system_version": system_version,
        "created_at": created_at,
        "status": status,
        "error": error,
        "error_category": error_category,
        "latency_seconds": latency_seconds,
        "answer": answer,
        "route": route,
        "retries": retries if retries is not None else 0,
        "support_status": support_status,
        "usefulness": usefulness,
        "used_web_search": int(bool(used_web_search)),
        "retrieval_available": int(bool(retrieval_available)),
        "retrieved_sources_json": _json(retrieved_sources or []),
        "retrieved_chunks_json": _json(retrieved_chunks or []),
        "used_sources_json": _json(used_sources or []),
        "final_evidence_json": _json(final_evidence or []),
        "grades_json": _json(grades or []),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_usage_available": int(bool(token_usage_available)),
        "estimated_cost_usd": estimated_cost_usd,
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def record_chat(
    request_id: str,
    question: str,
    thread_id: str,
    result: Optional[Dict[str, Any]],
    latency_seconds: float,
    error: Optional[BaseException] = None,
    path: Optional[Any] = None,
) -> None:
    """Persist the request and its Self-RAG evaluation. Never raises."""
    if not get_settings().live_eval_enabled:
        return
    try:
        now = _utcnow()
        if error is not None:
            request_record = _request_record(
                request_id,
                now,
                question,
                thread_id,
                status="failed",
                latency_seconds=latency_seconds,
                error=_sanitize_error(error),
                error_category=_classify_error(error),
            )
            evaluation_record = _evaluation_record(
                request_id,
                SELF_RAG_SYSTEM,
                now,
                status="failed",
                latency_seconds=latency_seconds,
                error=_sanitize_error(error),
                error_category=_classify_error(error),
            )
        else:
            result = result or {}
            request_record = _request_record(
                request_id,
                now,
                question,
                thread_id,
                status="ok",
                latency_seconds=latency_seconds,
                result=result,
            )
            prompt, completion, total, usage_available = _usage_from_result(result)
            sources = result.get("sources", [])
            evaluation_record = _evaluation_record(
                request_id,
                SELF_RAG_SYSTEM,
                now,
                status="ok",
                latency_seconds=latency_seconds,
                answer=result.get("answer", "") or "",
                route=result.get("route", "") or "",
                retries=_count_retries(result.get("trace", [])),
                support_status=result.get("support_status", "") or "",
                usefulness=result.get("usefulness", "") or "",
                used_web_search=bool(result.get("used_web_search")),
                retrieval_available=bool(result.get("retrieval_available")),
                retrieved_chunks=_chunk_artifacts(result.get("retrieved_chunks")),
                used_sources=_result_source_names(sources),
                final_evidence=_chunk_artifacts(result.get("final_evidence")),
                grades=_grade_artifacts(result.get("grades")),
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                token_usage_available=usage_available,
                estimated_cost_usd=estimate_cost(prompt, completion),
            )
        live_db.save_request(request_record, path)
        live_db.save_evaluation(evaluation_record, path)
        _refresh_auto_evaluation(request_id, SELF_RAG_SYSTEM, path)
    except Exception:  # noqa: BLE001 - logging must never break the chat request
        logger.exception("live request logging failed for request_id=%s", request_id)


# --------------------------------------------------------------------------- #
# Baseline RAG shadow evaluation
# --------------------------------------------------------------------------- #
def baseline_enabled() -> bool:
    settings = get_settings()
    return bool(
        getattr(settings, "live_eval_enabled", True)
        and getattr(settings, "live_eval_baseline_enabled", True)
    )


def schedule_baseline_shadow(background_tasks, request_id: str, question: str) -> bool:
    """Queue the baseline shadow evaluation to run after the response.

    Returns True when a task was queued. Never raises.
    """
    if not baseline_enabled():
        return False
    try:
        background_tasks.add_task(run_baseline_shadow, request_id, question)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("could not schedule baseline shadow for request_id=%s", request_id)
        return False


def run_baseline_shadow(request_id: str, question: str, path: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Run the baseline system for one request and persist its evaluation.

    Idempotent: a second invocation for the same ``(request_id, baseline_rag)``
    is a no-op. Returns the stored evaluation record (or None when skipped).
    """
    if not baseline_enabled():
        return None
    try:
        if live_db.has_evaluation(request_id, BASELINE_SYSTEM, path):
            return None
    except Exception:  # noqa: BLE001
        logger.exception("baseline dedupe check failed for request_id=%s", request_id)

    semaphore = _semaphore()
    semaphore.acquire()
    try:
        return _evaluate_baseline(request_id, question, path)
    finally:
        semaphore.release()


def _evaluate_baseline(request_id: str, question: str, path: Optional[Any] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    retrieved_sources: List[str] = []
    retrieved_chunks: List[Dict[str, Any]] = []
    retrieval_available = False
    raw_exc: Optional[BaseException] = None

    try:
        raw_docs = _get_retriever().invoke(question)
        retrieved_sources = _source_names(raw_docs)
        retrieved_chunks = _docs_to_chunks(raw_docs)
        retrieval_available = True
    except Exception as exc:  # noqa: BLE001 - recorded as retrieval-unavailable
        raw_exc = exc
        raw_docs = None

    # The shared raw retrieval is attached to the request so both systems are
    # scored against the identical top-k definition (mirrors the offline harness).
    try:
        live_db.set_request_retrieval(
            request_id, retrieved_sources, retrieval_available, path, retrieved_chunks=retrieved_chunks
        )
    except Exception:  # noqa: BLE001
        logger.exception("failed to attach raw retrieval for request_id=%s", request_id)

    created_at = _utcnow()
    status = "ok"
    error = None
    error_category = None
    answer = ""
    route = ""
    prompt = completion = total = None
    usage_available = False
    cost = None

    try:
        if raw_docs is None:
            assert raw_exc is not None
            raise raw_exc
        out = _get_baseline().run(question, raw_docs)
        answer = out.get("answer", "") or ""
        route = out.get("route", "")
        usage = out.get("usage") or {}
        if usage:
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            total = usage.get("total_tokens")
            usage_available = any(v is not None for v in (prompt, completion, total))
        cost = estimate_cost(prompt, completion)
    except Exception as exc:  # noqa: BLE001 - baseline failures never propagate
        status = "failed"
        error = _sanitize_error(exc)
        error_category = _classify_error(exc)
        logger.warning("baseline shadow failed for request_id=%s: %s", request_id, error)

    latency = round(time.perf_counter() - started, 3)
    record = _evaluation_record(
        request_id,
        BASELINE_SYSTEM,
        created_at,
        status=status,
        latency_seconds=latency,
        answer=answer,
        error=error,
        error_category=error_category,
        route=route,
        retries=0,
        retrieval_available=retrieval_available,
        retrieved_sources=retrieved_sources,
        retrieved_chunks=retrieved_chunks,
        used_sources=retrieved_sources,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        token_usage_available=usage_available,
        estimated_cost_usd=cost,
    )
    try:
        live_db.save_evaluation(record, path)
    except Exception:  # noqa: BLE001
        logger.exception("failed to persist baseline evaluation for request_id=%s", request_id)
    _refresh_auto_evaluation(request_id, BASELINE_SYSTEM, path)
    return record


def reset_runtime_state() -> None:
    """Test helper: drop cached baseline client and semaphore."""
    global _BASELINE, _SEMAPHORE
    with _BASELINE_LOCK:
        _BASELINE = None
    with _SEMAPHORE_LOCK:
        _SEMAPHORE = None
