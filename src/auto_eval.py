"""Deterministic, local automatic evaluation for live requests.

Everything here is computed from the artifacts recorded during a real request:
the question, the retrieved chunks (text, source, rank and score when the
retriever exposes one), the evidence a system actually used and the generated
answer. No model call, no network and no ground-truth labels are required.

Honest scope
------------
The measurements below are clearly-named **automatic proxies**, not ground
truth:

* ``query term coverage`` - how many content terms of the question appear in a
  retrieved chunk. A deterministic lexical proxy for retrieval relevance.
* ``answer relevance`` - question/answer content-term overlap.
* ``evidence support`` / ``unsupported claims`` - how much of the answer is
  actually backed by the recorded evidence (lexical grounding proxy).

Lexical overlap can miss semantic paraphrases and can be fooled by a chunk or
answer that merely repeats query words. It is reported as a proxy with its
limitations and is never presented as accuracy.

Conventional IR metrics - Precision@5, Recall@5, MRR, MAP@5 and nDCG@5 - require
a relevance reference (human judgments or a labelled ground truth). Because the
live system has neither automatically, those metrics are reported as
``unavailable`` with an explicit reason instead of being fabricated. This module
never assumes a retrieved chunk is relevant.

An optional, explicitly-configured LLM judge (:func:`judge_answer`) can add
semantic relevance/faithfulness scores. It is off by default, bounded to one
stored score per ``(request, system, method)`` and is never invoked by the
deterministic path or by the unit tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.config import get_settings


METHOD = "lexical_proxy_v1"
JUDGE_METHOD = "llm_judge_v1"
# The optional, explicitly-configured chunk relevance grader (baseline only).
GRADER_METHOD = "llm_grader_v1"

# Bumped when the stored automatic-evaluation shape changes. ``live_metrics``
# recomputes (upserting the same key, never duplicating) when a stored row was
# written by an older version.
METRICS_VERSION = 2

TRADITIONAL_METRIC_NAMES = (
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "map_at_5",
    "ndcg_at_5",
)

# Function words carry little retrieval/answer signal; excluding them stops the
# lexical overlap proxies from being inflated by "the/of/is".
STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into over
    under again further once here there when where why how all any both each few
    more most other some such no nor not only own same so than too very can will
    just should now is are was were be been being do does did doing have has had
    having i you he she it we they me him her us them my your his its our their
    what which who whom this that these those as about after before during while
    between out up down off above below please tell show give need want also may
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

LIMITATIONS_RETRIEVAL = (
    "Lexical query-term coverage is a deterministic proxy for retrieval relevance, "
    "not ground-truth relevance: it counts how many content terms of the question "
    "appear in the retrieved chunk text. Synonyms and paraphrases are missed, and a "
    "chunk can repeat query words without answering the question."
)
LIMITATIONS_ANSWER = (
    "Answer relevance and evidence support are deterministic lexical proxies, not "
    "semantic judgements: they measure term overlap between the answer, the question "
    "and the recorded evidence. A correct answer that paraphrases the evidence can "
    "score low and copying the evidence can score high while still being wrong, so "
    "these values do not prove correctness."
)
TRADITIONAL_METRICS_MESSAGE = (
    "Precision@5, Recall@5, MRR, MAP@5 and nDCG@5 require relevance judgments or a "
    "labelled ground-truth reference. The live system has no such reference "
    "automatically, so these metrics are reported as unavailable rather than "
    "invented."
)

# Grader-based ranking metrics use an automatic relevance grader (Self-RAG's own
# per-chunk grades, or the optional evaluator for Baseline RAG). They are real
# ranking metrics over the recorded grades, but the grader is a proxy: they are
# NOT human ground truth and must never be presented as accuracy.
GRADER_LIMITATIONS = (
    "Grader-based ranking metrics are computed from an automatic relevance grader "
    "(Self-RAG's recorded per-chunk grades, or the optional evaluator for Baseline "
    "RAG), not from human relevance judgments. The grader is a proxy, so these "
    "scores are not independent proof of correctness. Chunks that were never graded "
    "are excluded, never treated as not relevant."
)

GRADER_FORMULAS = {
    "precision_at_5": (
        "(# judged retrieved chunks in the top-5 that are relevant) / "
        "(# judged retrieved chunks in the top-5). The denominator is capped at 5 "
        "and never counts an ungraded chunk."
    ),
    "recall_at_5": (
        "(# relevant judged chunks in the top-5) / (total # judged-relevant chunks "
        "in the ranked list). Only computed when every ranked chunk was judged, so "
        "the denominator is a valid, well-defined judgment scope."
    ),
    "reciprocal_rank": (
        "1 / rank of the first judged-relevant chunk. Reported as 0.0 only when the "
        "full ranked list was judged and contains no relevant chunk; otherwise it is "
        "unavailable (an ungraded chunk earlier in the list could have been relevant)."
    ),
    "map_at_5": (
        "(sum over relevant ranks r<=5 of Precision@r) / min(#judged-relevant, 5). "
        "Requires a complete judgment of the ranked list."
    ),
    "ndcg_at_5": (
        "DCG@5 / IDCG@5 with binary gains (relevant=1) and a log2(rank+1) discount: "
        "DCG@5 = sum(gain_r / log2(r+1)); IDCG@5 = sum(1 / log2(i+1)) for "
        "i in 1..min(#judged-relevant, 5)."
    ),
}

_GRADER_METRIC_NAMES = (
    "precision_at_5",
    "recall_at_5",
    "reciprocal_rank",
    "map_at_5",
    "ndcg_at_5",
)


# --------------------------------------------------------------------------- #
# Optional judge schema (only used when the judge is explicitly configured)
# --------------------------------------------------------------------------- #
class JudgeOutput(BaseModel):
    answer_relevance: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    reason: str = ""


class ChunkGradeOutput(BaseModel):
    """Structured output of the optional, explicitly-configured chunk grader."""

    grades: List[bool] = Field(default_factory=list)
    reason: str = ""


# --------------------------------------------------------------------------- #
# Tokenization helpers
# --------------------------------------------------------------------------- #
def content_terms(text: Any) -> List[str]:
    """Lower-cased alphanumeric tokens with stop words removed (order kept)."""
    tokens = _TOKEN_RE.findall(str(text if text is not None else "").lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def _coverage(needle: Iterable[str], haystack: Iterable[str]) -> Optional[float]:
    """Fraction of distinct ``needle`` terms present in ``haystack`` (None if empty)."""
    needle_set = set(needle)
    if not needle_set:
        return None
    haystack_set = set(haystack)
    return round(len(needle_set & haystack_set) / len(needle_set), 4)


def _evidence_texts(evidence: Any) -> List[str]:
    """Normalize an evidence collection (dicts with ``text`` or plain strings)."""
    texts: List[str] = []
    for item in evidence or []:
        if isinstance(item, str):
            if item.strip():
                texts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("snippet") or ""
            if str(text).strip():
                texts.append(str(text))
    return texts


# --------------------------------------------------------------------------- #
# Retrieval proxy
# --------------------------------------------------------------------------- #
def normalize_source(value: Any) -> str:
    """Normalize a source/document identifier for comparison (basename, lower)."""
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    text = text.replace("\\", "/").rstrip("/")
    base = text.rsplit("/", 1)[-1] if "/" in text else text
    return base.strip().lower()


def chunk_identity(chunk_type: Any, source: Any, text: Any) -> str:
    """A stable identity for a retrieved chunk (type + source + text).

    The same helper is used when chunks are recorded and when they are graded, so
    a grade can be matched back to the exact chunk it judged - and a shared grade
    can be safely reused across systems only when both chunk ids and text match.
    """
    payload = f"{chunk_type or ''}|{source or ''}|{text or ''}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def normalize_chunks(chunks: Any) -> List[Dict[str, Any]]:
    """Normalize retrieved chunks to ordered ``{rank, source, title, text, score}``.

    Accepts either the persisted dictionaries or document-like objects. Rank
    defaults to the position in the list; a missing score stays ``None`` (never
    invented) so callers can tell when the retriever did not expose one. The
    recorded ``chunk_id``/``type``/``page`` are preserved when present so grade
    matching can use the exact chunk identity.
    """
    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(chunks or []):
        if isinstance(raw, dict):
            source = raw.get("source") or raw.get("document_name") or raw.get("title") or ""
            title = raw.get("title") or ""
            text = raw.get("text") or raw.get("snippet") or raw.get("content") or ""
            rank = raw.get("rank")
            score = raw.get("score")
            chunk_id = raw.get("chunk_id")
            chunk_type = raw.get("type")
            page = raw.get("page")
        else:
            metadata = getattr(raw, "metadata", {}) or {}
            source = metadata.get("document_name") or metadata.get("source") or metadata.get("title") or ""
            title = metadata.get("title") or ""
            text = getattr(raw, "page_content", "") or ""
            rank = getattr(raw, "rank", None)
            score = getattr(raw, "score", None)
            chunk_id = metadata.get("chunk_id")
            chunk_type = metadata.get("source_type")
            page = metadata.get("page")
        try:
            rank = int(rank) if rank is not None else index + 1
        except (TypeError, ValueError):
            rank = index + 1
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        normalized.append(
            {
                "rank": rank,
                "source": str(source or ""),
                "title": str(title or ""),
                "text": str(text or ""),
                "score": score,
                "chunk_id": str(chunk_id or ""),
                "type": str(chunk_type or ""),
                "page": page,
            }
        )
    normalized.sort(key=lambda chunk: chunk["rank"])
    return normalized


def evaluate_retrieval(question: Any, chunks: Any) -> Dict[str, Any]:
    """Deterministic retrieval-quality proxy over one system's retrieved chunks."""
    normalized = normalize_chunks(chunks)
    question_terms = set(content_terms(question))
    with_text = [chunk for chunk in normalized if chunk["text"].strip()]
    scores = [chunk["score"] for chunk in normalized if chunk["score"] is not None]

    coverages = []
    union_terms: set = set()
    for chunk in with_text:
        terms = set(content_terms(chunk["text"]))
        union_terms.update(terms)
        coverage = _coverage(question_terms, terms)
        if coverage is not None:
            coverages.append(coverage)

    block: Dict[str, Any] = {
        "method": METHOD,
        "metric": "lexical_query_term_coverage",
        "chunks_retrieved": len(normalized),
        "chunks_with_text": len(with_text),
        "distinct_sources": len({chunk["source"] for chunk in normalized if chunk["source"]}),
        "question_content_terms": len(question_terms),
        "mean_query_term_coverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
        "top_rank_query_term_coverage": (
            _coverage(question_terms, content_terms(with_text[0]["text"])) if with_text else None
        ),
        "query_terms_covered_by_any_chunk": _coverage(question_terms, union_terms) if with_text else None,
        "mean_retrieval_score": round(sum(scores) / len(scores), 6) if scores else None,
    }
    if not normalized:
        block.update(status="unavailable", reason="no_retrieved_chunks")
    elif not question_terms:
        block.update(status="insufficient_data", reason="question_has_no_content_terms")
    elif not with_text:
        block.update(status="insufficient_data", reason="chunk_text_not_recorded")
    else:
        block.update(status="ok", reason=None)
    block["limitations"] = LIMITATIONS_RETRIEVAL
    return block


# --------------------------------------------------------------------------- #
# Answer proxy
# --------------------------------------------------------------------------- #
def evaluate_answer(question: Any, answer: Any, evidence: Any) -> Dict[str, Any]:
    """Deterministic answer-relevance and evidence-grounding proxies."""
    answer_text = str(answer if answer is not None else "")
    answer_terms = set(content_terms(answer_text))
    question_terms = set(content_terms(question))
    evidence_terms: set = set()
    for text in _evidence_texts(evidence):
        evidence_terms.update(content_terms(text))

    claims = 0
    unsupported_claims = 0
    if evidence_terms:
        for sentence in _SENTENCE_SPLIT_RE.split(answer_text):
            sentence = sentence.strip()
            terms = set(content_terms(sentence))
            if not terms:
                continue
            claims += 1
            if not (terms & evidence_terms):
                unsupported_claims += 1

    unsupported = sorted(answer_terms - evidence_terms)
    block: Dict[str, Any] = {
        "method": METHOD,
        "answer_content_terms": len(answer_terms),
        "evidence_content_terms": len(evidence_terms),
        "answer_relevance": _coverage(question_terms, answer_terms) if answer_terms else None,
        "evidence_support": _coverage(answer_terms, evidence_terms) if evidence_terms else None,
        "claims_total": claims if evidence_terms else None,
        "unsupported_claims": unsupported_claims if evidence_terms else None,
        "unsupported_terms_total": len(unsupported),
        "unsupported_terms": unsupported[:25],
    }
    if not answer_text.strip():
        block.update(status="unavailable", reason="no_answer")
    elif not answer_terms:
        block.update(status="insufficient_data", reason="answer_has_no_content_terms")
    elif not evidence_terms:
        block.update(status="insufficient_data", reason="no_evidence_text_recorded")
    else:
        block.update(status="ok", reason=None)
    block["limitations"] = LIMITATIONS_ANSWER
    return block


# --------------------------------------------------------------------------- #
# Reported verification decisions (Self-RAG's own IsSUP/IsUSE outputs)
# --------------------------------------------------------------------------- #
def verification_block(evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Surface the recorded Self-RAG verification decisions as *reported* facts.

    These are the workflow's own self-assessment, not an independent correctness
    measurement, and are labelled as such.
    """
    evaluation = evaluation or {}
    return {
        "source": "self_rag_verification",
        "support_status": evaluation.get("support_status") or "",
        "usefulness": evaluation.get("usefulness") or "",
        "retries": evaluation.get("retries"),
        "used_web_search": bool(evaluation.get("used_web_search")),
        "route": evaluation.get("route") or "",
        "note": (
            "Reported Self-RAG verification decisions (IsSUP/IsUSE) recorded during "
            "the run. They are the workflow's own self-assessment, not ground-truth "
            "correctness."
        ),
    }


# --------------------------------------------------------------------------- #
# Conventional metrics that cannot be computed without a relevance reference
# --------------------------------------------------------------------------- #
def traditional_metrics_unavailable() -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "requires_relevance_judgments",
        "message": TRADITIONAL_METRICS_MESSAGE,
        "requires": "human relevance judgments or a labelled ground-truth reference",
    }
    for name in TRADITIONAL_METRIC_NAMES:
        block[name] = None
    return block


# --------------------------------------------------------------------------- #
# Grader-based ranking metrics (from recorded relevance grades)
# --------------------------------------------------------------------------- #
_MISSING = object()


def _grade_maps(grades: Any):
    """Index recorded grades by exact chunk id and by (type, source) fallback.

    The first recorded grade for a key wins, so an initial-retrieval grade is
    never overwritten by a later rewrite/web pass that retrieved the same chunk.
    """
    by_id: Dict[str, bool] = {}
    by_source: Dict[tuple, bool] = {}
    for grade in grades or []:
        if not isinstance(grade, dict):
            continue
        relevant = bool(grade.get("relevant"))
        chunk_id = str(grade.get("chunk_id") or "")
        if chunk_id and chunk_id not in by_id:
            by_id[chunk_id] = relevant
        key = (
            str(grade.get("type") or "internal"),
            normalize_source(grade.get("source") or grade.get("title") or ""),
        )
        if key not in by_source:
            by_source[key] = relevant
    return by_id, by_source


def _grade_for(chunk: Dict[str, Any], by_id: Dict[str, bool], by_source: Dict[tuple, bool]):
    chunk_id = str(chunk.get("chunk_id") or "")
    if chunk_id and chunk_id in by_id:
        return by_id[chunk_id]
    key = (
        str(chunk.get("type") or "internal"),
        normalize_source(chunk.get("source") or chunk.get("title") or ""),
    )
    if key in by_source:
        return by_source[key]
    return _MISSING


def grader_ranking_metrics(
    ranked_chunks: Any,
    grades: Any,
    *,
    source: str = "grader",
    k: int = 5,
) -> Dict[str, Any]:
    """Ranking metrics from an automatic grader's per-chunk relevance grades.

    ``ranked_chunks`` is the system's real ranked chunk sequence and ``grades``
    the relevance grades the grader actually produced. Grades are matched to
    chunks by exact ``chunk_id`` (falling back to ``(type, source)`` for records
    written before chunk ids existed). A chunk with no matching grade is
    *ungraded*: it is excluded from every denominator and never treated as not
    relevant.

    The five metrics (Precision@5, Recall@5, Reciprocal Rank, MAP@5, nDCG@5) are
    only computed when the recorded grades support a defensible calculation;
    otherwise a ``None`` value with an explicit ``reason`` is returned. The
    formulas live in :data:`GRADER_FORMULAS`.
    """
    ranked = normalize_chunks(ranked_chunks)
    by_id, by_source = _grade_maps(grades)

    flags: List[Optional[bool]] = []
    graded_total = 0
    relevant_total = 0
    for chunk in ranked:
        graded = _grade_for(chunk, by_id, by_source)
        if graded is _MISSING:
            flags.append(None)
        else:
            graded_total += 1
            if graded:
                relevant_total += 1
            flags.append(graded)

    total = len(ranked)
    ungraded_total = total - graded_total
    considered = flags[:k]
    judged_considered = [flag for flag in considered if flag is not None]
    relevant_considered = sum(1 for flag in judged_considered if flag)
    complete = total > 0 and ungraded_total == 0

    block: Dict[str, Any] = {
        "method": GRADER_METHOD,
        "source": source,
        "k": k,
        "gain": "binary (relevant=1, not relevant=0); ungraded chunks excluded",
        "ranked_total": total,
        "graded_total": graded_total,
        "ungraded_total": ungraded_total,
        "relevant_total_judged": relevant_total,
        "considered": min(k, total),
        "graded_considered": len(judged_considered),
        "relevant_considered": relevant_considered,
        "complete_judgment": complete,
        "precision_at_5": None,
        "recall_at_5": None,
        "reciprocal_rank": None,
        "map_at_5": None,
        "ndcg_at_5": None,
        "status": "insufficient_data",
        "reason": None,
        "metric_status": {},
        "formulas": GRADER_FORMULAS,
        "limitations": GRADER_LIMITATIONS,
    }
    metric_status = {name: "insufficient_data" for name in _GRADER_METRIC_NAMES}
    reasons: List[str] = []

    if total == 0:
        block["status"] = "unavailable"
        block["reason"] = "no_ranked_retrieval"
        block["metric_status"] = {name: "unavailable" for name in _GRADER_METRIC_NAMES}
        return block
    if graded_total == 0:
        block["status"] = "unavailable"
        block["reason"] = "no_grader_judgments"
        block["metric_status"] = {name: "unavailable" for name in _GRADER_METRIC_NAMES}
        return block

    # Precision@5 - honest denominator: judged chunks actually considered.
    if judged_considered:
        block["precision_at_5"] = round(relevant_considered / len(judged_considered), 4)
        metric_status["precision_at_5"] = "ok"

    # Recall@5 - requires a complete, well-defined judgment scope and >0 relevant.
    if complete and relevant_total > 0:
        block["recall_at_5"] = round(relevant_considered / relevant_total, 4)
        metric_status["recall_at_5"] = "ok"
    elif not complete:
        reasons.append("recall_at_5 requires every ranked chunk to be judged")
    else:
        reasons.append("recall_at_5 has no judged-relevant denominator")

    # Reciprocal Rank.
    first_relevant = next((i for i, flag in enumerate(considered, 1) if flag is True), None)
    if first_relevant is not None:
        earlier_ungraded = any(flag is None for flag in considered[: first_relevant - 1])
        if earlier_ungraded:
            reasons.append("reciprocal_rank uncertain: an earlier chunk is ungraded")
        else:
            block["reciprocal_rank"] = round(1.0 / first_relevant, 4)
            metric_status["reciprocal_rank"] = "ok"
    elif complete:
        block["reciprocal_rank"] = 0.0
        metric_status["reciprocal_rank"] = "ok"
    else:
        reasons.append("reciprocal_rank needs a complete judgment to report 0")

    # MAP@5.
    if complete and relevant_total > 0:
        running = 0
        ap_sum = 0.0
        for rank, flag in enumerate(considered, 1):
            if flag:
                running += 1
                ap_sum += running / rank
        block["map_at_5"] = round(ap_sum / min(relevant_total, k), 4)
        metric_status["map_at_5"] = "ok"
    elif not complete:
        reasons.append("map_at_5 requires every ranked chunk to be judged")

    # nDCG@5.
    if complete:
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, flag in enumerate(considered, 1)
            if flag
        )
        idcg = sum(
            1.0 / math.log2(i + 1) for i in range(1, min(relevant_total, k) + 1)
        )
        if idcg > 0:
            block["ndcg_at_5"] = round(dcg / idcg, 4)
            metric_status["ndcg_at_5"] = "ok"
        else:
            reasons.append("ndcg_at_5 has no judged-relevant gain to normalise")
    else:
        reasons.append("ndcg_at_5 requires every ranked chunk to be judged")

    values = [block[name] for name in _GRADER_METRIC_NAMES]
    if all(value is not None for value in values):
        block["status"] = "ok"
    elif any(value is not None for value in values):
        block["status"] = "partial"
    else:
        block["status"] = "insufficient_data"
    block["metric_status"] = metric_status
    block["reason"] = "; ".join(dict.fromkeys(reasons)) if reasons else None
    return block


def grader_metrics_not_evaluated(reason: str, message: str) -> Dict[str, Any]:
    """A grader block for a system whose chunks have not been graded."""
    return {
        "method": GRADER_METHOD,
        "source": "baseline_grader",
        "k": 5,
        "status": "not_evaluated",
        "reason": reason,
        "message": message,
        "gain": "binary (relevant=1, not relevant=0); ungraded chunks excluded",
        "ranked_total": 0,
        "graded_total": 0,
        "ungraded_total": 0,
        "relevant_total_judged": 0,
        "considered": 0,
        "graded_considered": 0,
        "relevant_considered": 0,
        "complete_judgment": False,
        "precision_at_5": None,
        "recall_at_5": None,
        "reciprocal_rank": None,
        "map_at_5": None,
        "ndcg_at_5": None,
        "metric_status": {name: "not_evaluated" for name in _GRADER_METRIC_NAMES},
        "formulas": GRADER_FORMULAS,
        "limitations": GRADER_LIMITATIONS,
    }


def same_ranked_retrieval(chunks_a: Any, chunks_b: Any) -> bool:
    """True when two ranked chunk lists are identical by chunk id and text.

    Used to decide whether a grader's judgments can be *reused* across systems:
    only when the exact same chunks (id + text + ordered rank) were retrieved.
    Document-name similarity alone is never sufficient.
    """
    a = normalize_chunks(chunks_a)
    b = normalize_chunks(chunks_b)
    if not a or not b or len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if (left["chunk_id"], left["type"], normalize_source(left["source"]), left["text"]) != (
            right["chunk_id"],
            right["type"],
            normalize_source(right["source"]),
            right["text"],
        ):
            return False
    return True


# --------------------------------------------------------------------------- #
# Grader note helpers
# --------------------------------------------------------------------------- #
def grader_reuse_note(other_system: str) -> str:
    return (
        f"{other_system} and this system retrieved the identical ranked chunks "
        "(same chunk ids and text), so the recorded relevance grades are reused "
        "for both. Grades are never copied based on document-name similarity."
    )


# --------------------------------------------------------------------------- #
# Per-system automatic evaluation
# --------------------------------------------------------------------------- #
def evaluate_system(
    question: Any,
    answer: Any,
    chunks: Any,
    evidence: Any = (),
    verification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One system's complete automatic evaluation for a single question."""
    return {
        "method": METHOD,
        "retrieval": evaluate_retrieval(question, chunks),
        "answer": evaluate_answer(question, answer, evidence),
        "evidence_chunks": len(_evidence_texts(evidence)),
        "verification": verification_block(verification),
        "traditional_metrics": traditional_metrics_unavailable(),
    }


# --------------------------------------------------------------------------- #
# Optional, explicitly-configured LLM judge (off by default, bounded)
# --------------------------------------------------------------------------- #
def judge_enabled(settings: Any = None) -> bool:
    """True only when the judge is explicitly enabled and a provider is configured."""
    try:
        settings = settings if settings is not None else get_settings()
    except Exception:  # noqa: BLE001 - a missing settings object means "disabled"
        return False
    if not getattr(settings, "live_eval_judge_enabled", False):
        return False
    return bool(getattr(settings, "groq_api_key", ""))


def judge_answer(
    question: Any,
    answer: Any,
    evidence: Any,
    settings: Any = None,
) -> Dict[str, Any]:
    """Run a single bounded LLM-as-judge evaluation. Raises on failure.

    Called only from the explicit judge endpoint when :func:`judge_enabled` is
    true. It never runs automatically and is never exercised by unit tests, so no
    live model call is made by the deterministic path.
    """
    from langchain_groq import ChatGroq

    settings = settings if settings is not None else get_settings()
    api_key = getattr(settings, "groq_api_key", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    evidence_text = "\n\n---\n\n".join(_evidence_texts(evidence)) or "(no evidence recorded)"
    llm = ChatGroq(
        api_key=api_key,
        model=getattr(settings, "groq_model", ""),
        temperature=0,
        max_tokens=600,
    ).with_structured_output(JudgeOutput, method="json_schema")

    prompt = (
        "You are an evaluation judge. Using ONLY the evidence below, score the "
        "answer. answer_relevance: how well the answer addresses the question "
        "(0-1). faithfulness: how well every claim in the answer is supported by "
        "the evidence (0-1); do not reward claims you cannot verify. List any "
        "claims that the evidence does not support in unsupported_claims, and give "
        "a one-line reason.\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence_text}"
    )
    result = llm.invoke(prompt)
    return {
        "method": JUDGE_METHOD,
        "question": str(question or ""),
        "answer_relevance": float(result.answer_relevance),
        "faithfulness": float(result.faithfulness),
        "unsupported_claims": list(result.unsupported_claims or []),
        "reason": result.reason or "",
    }


def grade_chunks(
    question: Any,
    chunks: Any,
    settings: Any = None,
) -> List[Dict[str, Any]]:
    """Grade one system's retrieved chunks for relevance with a single LLM call.

    Optional and explicitly configured only (the caller gates it with
    :func:`judge_enabled`). Bounded to one call per invocation, never run
    automatically and never exercised by unit tests. Returns one grade entry per
    chunk that the model returned a decision for; any chunk without a decision is
    simply omitted and stays *ungraded* downstream - it is never assumed
    irrelevant.
    """
    from langchain_groq import ChatGroq

    settings = settings if settings is not None else get_settings()
    api_key = getattr(settings, "groq_api_key", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    normalized = normalize_chunks(chunks)
    if not normalized:
        return []

    listing = "\n\n".join(
        f"[{i}] {(chunk['text'] or '(no text recorded)')[:1500]}"
        for i, chunk in enumerate(normalized, 1)
    )
    llm = ChatGroq(
        api_key=api_key,
        model=getattr(settings, "groq_model", ""),
        temperature=0,
        max_tokens=600,
    ).with_structured_output(ChunkGradeOutput, method="json_schema")
    prompt = (
        "You are a strict relevance grader. For each retrieved chunk decide "
        "whether it contains information useful for answering the question. "
        "Return EXACTLY one boolean per chunk, in order, in grades. A chunk with "
        "no usable text is not relevant. Do not guess.\n\n"
        f"Question:\n{question}\n\nChunks:\n{listing}"
    )
    result = llm.invoke(prompt)

    grades: List[Dict[str, Any]] = []
    for index, chunk in enumerate(normalized):
        if index >= len(result.grades):
            break
        grades.append(
            {
                "chunk_id": chunk["chunk_id"],
                "type": chunk["type"] or "internal",
                "source": chunk["source"],
                "title": chunk["title"],
                "relevant": bool(result.grades[index]),
            }
        )
    return grades
