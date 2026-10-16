"""Run the Axiom evaluation.

Compares two systems on the same question set:

  1. baseline_rag    - the existing Pinecone retriever -> existing Groq model ->
                       answer. No Self-RAG graph, no grading, no retries, no web
                       fallback.
  2. axiom_self_rag  - the existing LangGraph Self-RAG workflow, unchanged.

The baseline deliberately does NOT import or invoke anything from
``src.self_rag``. It only reuses the configured retriever (``src.vectorstore``),
the configured Groq model, and the settings, so the comparison stays fair.

Usage (from the project root)::

    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --validate-only
    python -m evaluation.run_evaluation --systems baseline --limit 3

This script never prints API keys. Per-question results are written to
``evaluation/results/`` as CSV and JSON. Human-scored columns are left blank
until reviewed (see evaluation/README.md and evaluation/metrics.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Make the project importable whether run as a script or as a module. -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: src.config / src.vectorstore / src.self_rag are imported lazily inside
# the functions that need them so that ``--validate-only`` works even when the
# runtime dependencies (pinecone, langgraph, langchain) are not installed.

EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVALUATION_DIR / "dataset.json"
DEFAULT_OUT_DIR = EVALUATION_DIR / "results"

try:  # package import (python -m evaluation.run_evaluation)
    from evaluation.manual_scores import (
        MANUAL_SCORES_FILENAME,
        apply_manual_scores,
        collect_manual_scores,
        load_manual_scores,
        persist_manual_scores,
    )
except ImportError:  # standalone script (python evaluation\run_evaluation.py)
    from manual_scores import (  # type: ignore
        MANUAL_SCORES_FILENAME,
        apply_manual_scores,
        collect_manual_scores,
        load_manual_scores,
        persist_manual_scores,
    )

BASELINE = "baseline_rag"
AXIOM = "axiom_self_rag"
ALL_SYSTEMS = (BASELINE, AXIOM)

VALID_CATEGORIES = {"direct", "paraphrased", "multi_document", "ambiguous", "unanswerable"}

# Ordered column layout for the per-question CSV. Human-scored columns are the
# last three and stay blank until a reviewer fills them in.
RESULT_FIELDS = [
    "run_id",
    "system_version",
    "question_id",
    "category",
    "question",
    "answer",
    # Raw top-k retrieval (identical definition for both systems). This is what
    # the retrieval metrics score. ``retrieval_available`` is False when the raw
    # retrieval could not be recorded for a question (never scored as zero).
    "retrieved_sources",
    "retrieved_source_count",
    "retrieval_available",
    # Documents the system actually used (after Self-RAG grading / query
    # rewriting / web search). Never compared against runbook filenames.
    "used_sources",
    "used_source_count",
    "ground_truth_sources",
    "expected_answer_points",
    "latency_seconds",
    "retries",
    "route",
    "support_status",
    "usefulness",
    "used_web_search",
    "request_failed",
    "error",
    "error_category",
    # Automatic usage data; blank/"false" when the provider did not return it.
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_usage_available",
    # ---- Human-evaluated fields: leave blank until reviewed. ----
    "answer_correctness_manual",
    "faithfulness_manual",
    "manual_notes",
]

# Trace markers that indicate a Self-RAG retry/rewrite cycle for the retry count.
RETRY_MARKERS = (
    "Rewrote internal query",
    "Prepared internet search query",
    "Revised answer for grounding",
)


# --------------------------------------------------------------------------- #
# Dataset loading / validation
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} must be a JSON list of question objects.")
    return data


def validate_dataset(items: List[Dict[str, Any]], known_sources: Optional[set] = None) -> List[str]:
    """Return a list of human-readable validation errors (empty means valid)."""
    errors: List[str] = []
    if len(items) != 30:
        errors.append(f"Expected exactly 30 questions, found {len(items)}.")

    seen_ids, seen_questions = set(), {}
    for i, item in enumerate(items):
        where = f"item[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{where} is not an object.")
            continue

        qid = item.get("id")
        if not isinstance(qid, str) or not qid.strip():
            errors.append(f"{where} is missing a non-empty string 'id'.")
            qid = where
        elif qid in seen_ids:
            errors.append(f"Duplicate id '{qid}'.")
        else:
            seen_ids.add(qid)

        q = item.get("question")
        if not isinstance(q, str) or len(q.strip()) < 8:
            errors.append(f"{qid}: 'question' must be a string of at least 8 characters.")
        else:
            norm = q.strip().lower()
            if norm in seen_questions:
                errors.append(f"{qid}: duplicate question text (same as {seen_questions[norm]}).")
            else:
                seen_questions[norm] = qid

        cat = item.get("category")
        if cat not in VALID_CATEGORIES:
            errors.append(f"{qid}: category '{cat}' is not one of {sorted(VALID_CATEGORIES)}.")

        pts = item.get("expected_answer_points")
        if not isinstance(pts, list) or not pts:
            errors.append(f"{qid}: 'expected_answer_points' must be a non-empty list.")

        sources = item.get("relevant_sources")
        if not isinstance(sources, list):
            errors.append(f"{qid}: 'relevant_sources' must be a list (may be empty).")
        elif known_sources is not None:
            unknown = [s for s in sources if s not in known_sources]
            if unknown:
                errors.append(f"{qid}: unknown relevant_sources {unknown}. Known: {sorted(known_sources)}.")

        if not isinstance(item.get("answerable"), bool):
            errors.append(f"{qid}: 'answerable' must be a boolean.")

        # An answerable question with ground-truth sources is required for
        # retrieval metrics; unanswerable questions intentionally have none.
        if item.get("answerable") and not sources:
            errors.append(f"{qid}: answerable questions must list at least one relevant_source.")

    return errors


def discover_known_sources() -> set:
    docs_dir = PROJECT_ROOT / "documents"
    if not docs_dir.exists():
        return set()
    return {p.name for p in docs_dir.iterdir() if p.is_file()}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _secret_values() -> List[str]:
    from src.config import get_settings

    s = get_settings()
    return [v for v in (s.groq_api_key, s.pinecone_api_key, s.tavily_api_key) if v]


def sanitize(text: str, secrets: List[str]) -> str:
    """Remove any accidental secret from a string before writing it to disk."""
    if not text:
        return ""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


# Error buckets used to separate provider/quota problems from application bugs and
# structured-output parse failures. Keep this list small and stable; metrics.py
# reports counts per bucket.
ERROR_QUOTA = "quota_or_rate_limit"
ERROR_PARSER = "parser"
ERROR_AUTH = "auth"
ERROR_NETWORK = "network"
ERROR_APPLICATION = "application"


def classify_error(exc: BaseException) -> str:
    """Classify an exception into a coarse bucket for reporting.

    Uses the exception's MRO names and message text so it works whether the
    provider error is a ``groq``/``openai`` class or wrapped by LangChain.
    """
    names = {cls.__name__.lower() for cls in type(exc).__mro__}
    name = type(exc).__name__.lower()
    text = str(exc).lower()

    quota_markers = ("ratelimit", "rate limit", "quota", "429", "too many requests", "tpm", "tpd")
    if any(m in name for m in ("ratelimit", "quota")) or any(m in text for m in quota_markers):
        return ERROR_QUOTA

    if any(m in name for m in ("outputparser", "validationerror", "jsondecode", "structuredoutput")):
        return ERROR_PARSER
    if "parse" in text and "json" in text:
        return ERROR_PARSER

    if any(m in name for m in ("authentication", "permission")) or any(
        m in text for m in ("api key", "401", "403", "permission")
    ):
        return ERROR_AUTH

    if any(m in name for m in ("connection", "timeout", "connecterror")) or any(
        m in text for m in ("timed out", "connection", "temporarily unavailable")
    ):
        return ERROR_NETWORK

    return ERROR_APPLICATION


def _doc_source_name(metadata: Dict[str, Any]) -> str:
    meta = metadata or {}
    name = meta.get("document_name")
    if name:
        return str(name)
    src = meta.get("source")
    if src:
        return Path(str(src)).name
    return str(meta.get("title") or "")


def retrieved_source_names(docs: List[Any]) -> List[str]:
    names: List[str] = []
    for d in docs or []:
        name = _doc_source_name(getattr(d, "metadata", {}) or {})
        if name:
            names.append(name)
    return names


def _get_shared_retriever():
    """The single raw retriever used to score retrieval quality for both systems."""
    from src.vectorstore import get_retriever

    return get_retriever()


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    """Read a previously written per-question CSV/JSON file (for score recovery)."""
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def format_context(docs: List[Any]) -> str:
    blocks = []
    for i, d in enumerate(docs or [], 1):
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("title") or meta.get("document_name") or meta.get("source") or f"Document {i}"
        blocks.append(f"[Doc {i}] {title}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def extract_usage(message: Any) -> Optional[Dict[str, Optional[int]]]:
    """Best-effort token usage. Returns None when the provider gave none.

    Never synthesises numbers: if no usage metadata is present, this returns
    None and the caller records blanks.
    """
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    rm = getattr(message, "response_metadata", None) or {}
    tu = rm.get("token_usage") if isinstance(rm, dict) else None
    if isinstance(tu, dict) and tu:
        return {
            "prompt_tokens": tu.get("prompt_tokens"),
            "completion_tokens": tu.get("completion_tokens"),
            "total_tokens": tu.get("total_tokens"),
        }
    return None


def count_axiom_retries(trace: List[str]) -> int:
    return sum(1 for t in trace or [] if any(marker in t for marker in RETRY_MARKERS))


def _blank_row(run_id: str, system: str, item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "system_version": system,
        "question_id": item["id"],
        "category": item.get("category", ""),
        "question": item["question"],
        "answer": "",
        "retrieved_sources": "",
        "retrieved_source_count": 0,
        "retrieval_available": False,
        "used_sources": "",
        "used_source_count": 0,
        "ground_truth_sources": ";".join(item.get("relevant_sources", [])),
        "expected_answer_points": " | ".join(item.get("expected_answer_points", [])),
        "latency_seconds": "",
        "retries": "",
        "route": "",
        "support_status": "",
        "usefulness": "",
        "used_web_search": "",
        "request_failed": False,
        "error": "",
        "error_category": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "total_tokens": "",
        "token_usage_available": False,
        "answer_correctness_manual": "",
        "faithfulness_manual": "",
        "manual_notes": "",
    }


# --------------------------------------------------------------------------- #
# Baseline (traditional RAG) — independent of the Self-RAG graph
# --------------------------------------------------------------------------- #
class BaselineRAG:
    """Retrieved context -> single Groq generation. No grading, no retries.

    Retrieval itself is performed once per question by the runner and passed in,
    so both systems are scored against the identical raw top-k retrieval.
    """

    SYSTEM_PROMPT = (
        "You are an enterprise cloud operations and incident-response assistant. "
        "Answer the question using only the supplied evidence. If the evidence does "
        "not contain the answer, say so instead of inventing details."
    )

    def __init__(self) -> None:
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq
            from src.config import get_settings

            s = get_settings()
            if not s.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            # Same configured model as Axiom; kept independent of src.self_rag.
            self._llm = ChatGroq(api_key=s.groq_api_key, model=s.groq_model, temperature=0)
        return self._llm

    def run(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        context = format_context(docs)
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", self.SYSTEM_PROMPT), ("human", "Question:\n{question}\n\nEvidence:\n{context}")]
        )
        message = self._get_llm().invoke(
            prompt.format_messages(question=question, context=context)
        )
        usage = extract_usage(message)
        return {
            "answer": getattr(message, "content", "") or "",
            "route": "internal_retrieval",
            "retries": 0,
            "usage": usage,
        }


# --------------------------------------------------------------------------- #
# Axiom (complete Self-RAG) — the existing workflow, unchanged
# --------------------------------------------------------------------------- #
def run_axiom(question: str, thread_id: str) -> Dict[str, Any]:
    # Imported lazily so baseline-only runs do not need the graph to initialise.
    from src.self_rag import run_self_rag

    result = run_self_rag(question, thread_id)
    sources = [
        _doc_source_name({"document_name": s.get("title"), "source": s.get("source"), "title": s.get("title")})
        for s in result.get("sources", [])
    ]
    sources = [s for s in sources if s]
    return {
        "answer": result.get("answer", ""),
        # What Axiom actually used (post-grading / rewrite / web). Kept separate
        # from the shared raw retrieval that the retrieval metrics score.
        "used_sources": sources,
        "route": result.get("route", ""),
        "retries": count_axiom_retries(result.get("trace", [])),
        "support_status": result.get("support_status", ""),
        "usefulness": result.get("usefulness", ""),
        "used_web_search": bool(result.get("used_web_search")),
        "usage": None,  # run_self_rag does not expose token usage
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _record_usage(row: Dict[str, Any], usage: Optional[Dict[str, Optional[int]]]) -> None:
    if not usage:
        return
    row["prompt_tokens"] = usage.get("prompt_tokens") if usage.get("prompt_tokens") is not None else ""
    row["completion_tokens"] = usage.get("completion_tokens") if usage.get("completion_tokens") is not None else ""
    row["total_tokens"] = usage.get("total_tokens") if usage.get("total_tokens") is not None else ""
    row["token_usage_available"] = any(
        usage.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


def evaluate(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    secrets = _secret_values()

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 2

    items = load_dataset(dataset_path)
    known_sources = discover_known_sources()
    errors = validate_dataset(items, known_sources)
    if errors:
        print("Dataset validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"Dataset OK: {len(items)} questions across {len(set(i['category'] for i in items))} categories.")

    if args.validate_only:
        return 0

    systems = ALL_SYSTEMS if args.systems == "both" else (args.systems,)
    if args.limit:
        items = items[: args.limit]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.config import get_settings

    s = get_settings()
    print("-" * 62)
    print(f"Run id        : {run_id}")
    print(f"Systems       : {', '.join(systems)}")
    print(f"Questions     : {len(items)}")
    print(f"Groq model    : {s.groq_model}")
    print(f"Pinecone index: {s.pinecone_index_name} / namespace {s.pinecone_namespace} (top_k={s.top_k})")
    print(f"Output dir    : {out_dir}")
    print("-" * 62)

    baseline = BaselineRAG() if BASELINE in systems else None
    retriever = None

    rows: List[Dict[str, Any]] = []
    for item in items:
        qid = item["id"]
        question = item["question"]

        # Raw top-k retrieval, performed once per question and shared by both
        # systems, so Precision@5 / Recall@5 use the identical retrieval
        # definition. It does not depend on whether a system's request later
        # fails, so downstream quota/parser failures cannot bias retrieval scores.
        raw_docs: Optional[List[Any]] = None
        raw_exc: Optional[BaseException] = None
        try:
            if retriever is None:
                retriever = _get_shared_retriever()
            raw_docs = retriever.invoke(question)
        except Exception as exc:  # noqa: BLE001 - recorded as retrieval-unavailable
            raw_exc = exc
        raw_sources = retrieved_source_names(raw_docs) if raw_docs is not None else []

        for system in systems:
            row = _blank_row(run_id, system, item)
            row["retrieved_sources"] = ";".join(raw_sources)
            row["retrieved_source_count"] = len(raw_sources)
            row["retrieval_available"] = raw_docs is not None
            if raw_docs is not None:
                row["used_sources"] = ";".join(raw_sources)
                row["used_source_count"] = len(raw_sources)
            started = time.perf_counter()
            try:
                if system == BASELINE:
                    assert baseline is not None
                    if raw_docs is None:
                        assert raw_exc is not None
                        raise raw_exc
                    out = baseline.run(question, raw_docs)
                    row["answer"] = out.get("answer", "")
                    row["route"] = out.get("route", "")
                    row["retries"] = out.get("retries", 0)
                    _record_usage(row, out.get("usage"))
                else:
                    thread_id = f"eval-{run_id}-{qid}"[:120]
                    out = run_axiom(question, thread_id)
                    used = out.get("used_sources", [])
                    row["answer"] = out.get("answer", "")
                    row["used_sources"] = ";".join(used)
                    row["used_source_count"] = len(used)
                    row["route"] = out.get("route", "")
                    row["retries"] = out.get("retries", 0)
                    row["support_status"] = out.get("support_status", "")
                    row["usefulness"] = out.get("usefulness", "")
                    row["used_web_search"] = out.get("used_web_search", "")
                    _record_usage(row, out.get("usage"))
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - keep the experiment running
                row["request_failed"] = True
                row["error"] = sanitize(f"{type(exc).__name__}: {exc}", secrets)
                row["error_category"] = classify_error(exc)
                status = f"FAILED ({row['error_category']})"
            finally:
                row["latency_seconds"] = round(time.perf_counter() - started, 3)
            rows.append(row)
            print(f"[{qid}] {system:<15} {status} ({row['latency_seconds']}s)")
            if args.sleep:
                time.sleep(args.sleep)

    # --- Preserve manual scores across reruns -------------------------------
    # The durable store is keyed by (run_id, system_version, question_id), so a
    # regenerated file with blank manual columns never erases reviewed scores.
    manual_path = out_dir / MANUAL_SCORES_FILENAME
    store = load_manual_scores(manual_path)
    prior = out_dir / f"per_question_{run_id}.csv"
    if prior.exists():
        try:
            store = {**store, **collect_manual_scores(_read_rows(prior))}
        except Exception:  # noqa: BLE001 - a corrupt prior file must not abort the run
            pass
    restored = apply_manual_scores(rows, store)
    persisted = persist_manual_scores(manual_path, rows)

    csv_path = out_dir / f"per_question_{run_id}.csv"
    json_path = out_dir / f"per_question_{run_id}.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)
    latest = out_dir / "per_question_latest.csv"
    _write_csv(latest, rows)

    failures = sum(1 for r in rows if r["request_failed"])
    print("-" * 62)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Also wrote {latest}")
    print(f"Rows: {len(rows)} | failures: {failures}")
    if persisted:
        print(f"Manual score store: {manual_path} ({persisted} scored row(s) kept)")
    if restored:
        print(f"Restored {restored} previously reviewed score(s) for this run id.")
    print("Next: score the manual columns, then run: python evaluation/metrics.py")
    print("Scores are preserved keyed by run id; reuse --run-id to keep them on a rerun.")
    return 0 if failures == 0 else 3


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Axiom baseline vs Self-RAG evaluation.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to dataset.json")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for result files")
    parser.add_argument(
        "--systems",
        choices=["both", BASELINE, AXIOM],
        default="both",
        help="Which system(s) to run.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N questions (smoke test).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between runs (rate limiting).")
    parser.add_argument("--run-id", default="", help="Override the generated run id.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the dataset and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
