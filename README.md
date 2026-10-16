# Axiom - SelfRAG Copilot

A production-style **Self-RAG** for operations and incident response. It searches private operational knowledge in Pinecone first, self-grades the retrieved evidence, corrects weak retrieval/generation, and uses **internet search only when the private knowledge base is insufficient**.

The project also demonstrates **LangGraph SQLite persistence memory** so follow-up questions can reuse the same incident context through a stable `thread_id`.

## Architecture

```text
User Incident / Follow-up
        ↓
LangGraph SQLite Memory
        ↓
Contextualize Follow-up
        ↓
Decide Retrieval
        ↓
Private Pinecone Knowledge Base
        ↓
Grade Retrieved Documents
   ┌────┴───────────────┐
Relevant             Weak / Missing
   ↓                      ↓
Generate              Rewrite Query
   ↓                      ↓
IsSUP              Retry Private KB
   ↓                      ↓
Revise if needed   Internet Search Fallback
   ↓                      ↓
IsUSE              Grade Web Evidence
   ↓                      ↓
Final Answer ← Generate → IsSUP → IsUSE
        ↓
SQLite Checkpoint / Memory
```

## Tech stack

- **LangGraph** — Self-RAG workflow, conditional routing, persistent thread state
- **Groq `openai/gpt-oss-120b`** — routing, query rewriting, generation, relevance grading, IsSUP and IsUSE
- **Hugging Face `BAAI/bge-small-en-v1.5`** — embeddings for private operational documents
- **Pinecone** — private runbooks/SOPs/postmortems knowledge base
- **Tavily** — controlled internet-search fallback
- **SQLite** — LangGraph persistence memory + simple audit database for the demo
- **FastAPI** — application/API backend
- **HTML/CSS/JavaScript** — incident-command-center UI with document upload
- **Docker** — deployment packaging

## Project structure

```text
CloudOps-Sentinel-Enterprise-Incident-Response-Self-RAG-Copilot/
├── app.py
├── data_ingestion.py          
├── src/
│   ├── config.py
│   ├── db.py
│   ├── ingestion.py
│   ├── models.py
│   ├── self_rag.py
│   └── vectorstore.py
├── documents/                
│   ├── checkout-api-runbook.md
│   ├── payments-high-cpu-runbook.md
│   └── deployment-rollback-sop.md
├── templates/index.html
├── static/styles.css
├── static/app.js
├── uploads/                  
├── data/                      
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

# Key Features

Self-RAG workflow with retrieval and generation verification

Private-first retrieval using Pinecone

Evidence grading before generating answers

IsSUP — verifies whether the answer is supported by evidence

IsUSE — evaluates answer usefulness

Query rewriting and retrieval retry for weak results

Tavily web fallback when private knowledge is insufficient

LangGraph SQLite memory for contextual follow-up questions

Document upload for adding new knowledge to the system

FastAPI backend with a simple web interface

LangSmith tracing for workflow observability

Activate venv ->  .\.venv\Scripts\Activate.ps1


## Live Evaluation Dashboard

The dashboard (`Evaluation Dashboard` in the sidebar) shows **only a paginated,
newest-first question history**. There are no aggregate KPI cards, comparison
panels, workflow charts or daily trends — clicking a question opens a detail view
scoped exclusively to that request's own records. Every value is computed from the
recorded execution artifacts; nothing is read from the offline benchmark and no
number is hard-coded.

### Runtime flow

1. `POST /api/chat` runs Self-RAG and returns the normal answer (with a
   `request_id`).
2. The request record + the Self-RAG evaluation are written to the shared SQLite
   database, including the real execution artifacts (retrieved chunk text/ranks,
   the final evidence actually used, the relevance grades). Logging is fail-safe: a
   logging error never breaks the chat.
3. After the response is sent, FastAPI `BackgroundTasks` runs the Baseline RAG
   shadow evaluation (raw top-k retrieval → one Groq generation).
4. For each system the **automatic evaluation** is computed locally from those
   artifacts and persisted once (see below). Opening a question reuses the stored
   evaluation — it never re-runs answer generation and never creates duplicates.

### Storage (reuses the existing `data/audit.db`)

Added tables: `live_requests`, `live_evaluations` (unique per `request_id` +
`system_version`), `live_quality_scores`, `live_auto_evaluations` (unique per
`request_id` + `system_version` + `method`), and the legacy
`live_relevance_labels` table (retained only so previously stored data is never
lost — the live evaluation no longer reads it). Tables/indexes are created
automatically on startup by `init_live_db()`; columns added after a table's first
release (e.g. `retrieved_chunks_json`) are applied additively with
`ALTER TABLE ADD COLUMN`, so existing rows and data are preserved with no separate
migration tool.

### Resetting live evaluation data

Before a fresh testing session, clear the recorded live data (the live-evaluation
tables only) with the confirmation-gated reset script:

```bash
python reset_live_eval.py --confirm
```

It deletes rows in dependency order (`live_auto_evaluations` →
`live_relevance_labels` → `live_quality_scores` → `live_evaluations` →
`live_requests`) inside a single transaction, prints the deleted row count per
table, then verifies the tables are empty. Without `--confirm` it deletes nothing
and exits `2`.

Preserved: the database file, schema and indexes, the `rag_audit` log, LangGraph
conversation memory, uploaded documents, Pinecone vectors,
`evaluation/dataset.json`, and the offline benchmark results in
`evaluation/results/`.

### API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/evaluation/history` | paginated per-question history (the dashboard's main view) |
| `GET /api/evaluation/requests/{id}` | one question's full evaluation: both answers, retrieved chunks, final evidence, grading, grader-based ranking metrics, automatic proxies, verification, latency/tokens/errors |
| `POST /api/evaluation/requests/{id}/judge?system=...` | run the optional, explicitly-configured LLM evaluator once (bounded; `503` when not configured) |
| `POST /api/evaluation/requests/{id}/grade-baseline` | grade Baseline RAG's chunks once with the optional evaluator (bounded; `503` when not configured) |
| `GET /api/evaluation/judge/status` | whether the optional evaluator is configured (no model call) |
| `GET /api/evaluation/summary` | aggregate counts/latency/tokens (retained for the API; not shown on the dashboard) |
| `GET /api/evaluation/comparison` | per-system aggregate blocks (retained; not shown on the dashboard) |
| `GET /api/evaluation/daily` | per-day trend (retained; not shown on the dashboard) |

All accept `date_from` / `date_to` (`YYYY-MM-DD`); `history` also accepts `page`,
`page_size`, `status` and `system`. No endpoint runs answer generation, and the
deterministic automatic evaluation never calls a model.

### Automatic question-level evaluation (no labels, no benchmark)

Each question is evaluated separately for **Baseline RAG** and **Self-RAG**
directly from what happened, using `src/auto_eval.py`:

- **Retrieval quality (proxy):** how many content terms of the *question* appear
  in each retrieved chunk — mean query-term coverage, top-rank coverage, union
  coverage, chunk/source counts and the mean retrieval score *when the retriever
  exposed one*. Baseline RAG is scored on its retrieved context; Self-RAG is scored
  on its **final evidence** (kept distinct from its initial retrieval).
- **Answer quality (proxy):** answer relevance (question↔answer term overlap),
  evidence support (answer terms backed by the recorded evidence), and unsupported
  claim/term counts.
- **Reported verification:** the Self-RAG workflow's own IsSUP/IsUSE decisions,
  retries and web-search activity, labelled as a self-assessment.

These are **clearly-named lexical proxies**, not ground truth — each carries a
short methodology/limitations note in the UI. A retrieved document is never
assumed relevant, and no value is fabricated or presented as `0` when it is
actually unavailable.

**Metrics that cannot be validly computed.** Precision@5, Recall@5, MRR, MAP@5 and
nDCG@5 *as ground-truth* metrics require human relevance judgments or a labelled
ground-truth reference. The live system has neither automatically, so it reports
them as `unavailable` with an explicit reason instead of inventing a score. (They
remain computed for the offline benchmark, which does have a labelled dataset.)

### Grader-based ranking metrics (from recorded grades)

Separately from the ground-truth block, the live dashboard computes real ranking
metrics from an **automatic relevance grader** — Self-RAG's own recorded per-chunk
grades, and (when available) the optional evaluator's grades for Baseline RAG.
These are ranking metrics over recorded grades, but the grader is a proxy, so they
are labelled as such and kept distinct from the ground-truth metrics.

For each system the metrics are computed over that system's **real ranked chunk
sequence** and the grades that were actually recorded. A chunk with no recorded
grade is *ungraded*: it is excluded from every denominator and never treated as
not relevant.

| Metric | Formula (and when it is computed) |
| --- | --- |
| Precision@5 | judged top-5 chunks that are relevant / judged top-5 chunks (denominator capped at 5, never counts an ungraded chunk) |
| Recall@5 | relevant judged top-5 / total judged-relevant in the ranked list — only when **every** ranked chunk was judged (valid, well-defined scope) and the denominator is > 0 |
| Reciprocal Rank | `1 / rank` of the first judged-relevant chunk; `0.0` **only** when the whole ranked list was judged and contains nothing relevant |
| MAP@5 | `(sum over relevant ranks r<=5 of Precision@r) / min(#judged-relevant, 5)` — requires a complete judgment |
| nDCG@5 | `DCG@5 / IDCG@5`, binary gains, `log2(r+1)` discount (see the offline table below); requires a complete judgment and > 0 judged-relevant |

When the recorded grades do not support a defensible calculation, each metric
returns `insufficient_data` (or `unavailable`) with a reason instead of a number.

**Baseline RAG grade isolation.** Baseline grades are obtained in exactly two
ways and never copied by document-name similarity: (1) when Baseline RAG and
Self-RAG retrieved the **identical ranked chunks** (same chunk ids *and* text),
Self-RAG's recorded grades are reused after verifying that identity; (2)
otherwise, grades produced by the optional, explicitly-configured baseline chunk
grader are used. If neither applies, the block is `not_evaluated` with a reason.
No extra model call is made by default — the grader is off unless explicitly
configured, exactly like the optional LLM evaluator.

### Optional LLM evaluator

Off by default. When `LIVE_EVAL_JUDGE_ENABLED=true` **and** a Groq key is
configured, `POST /api/evaluation/requests/{id}/judge` runs one bounded
LLM-as-judge call (answer relevance + faithfulness against the recorded evidence)
and stores the result under method `llm_judge_v1`. It is bounded to one stored
score per `(request, system)` — a repeat call returns the stored score without
another model call — and evaluator failures are reported as failures, never as
scores. Unit tests never enable it, so no live calls happen in the test suite.

### Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `LIVE_EVAL_ENABLED` | `true` | master switch for runtime logging |
| `LIVE_EVAL_BASELINE_ENABLED` | `true` | run the Baseline RAG shadow evaluation |
| `LIVE_EVAL_MAX_CONCURRENCY` | `2` | max concurrent baseline evaluations |
| `LIVE_EVAL_JUDGE_ENABLED` | `false` | enable the optional LLM evaluator endpoint |
| `GROQ_INPUT_COST_PER_1M` | unset | USD/1M prompt tokens (cost estimate) |
| `GROQ_OUTPUT_COST_PER_1M` | unset | USD/1M completion tokens (cost estimate) |

Set `LIVE_EVAL_BASELINE_ENABLED=false` to stop the extra Groq call entirely.
Estimated cost is `null` unless both pricing vars are set.

### Metric availability

- **Measured automatically (no model, no labels):** per-question retrieval proxies
  (chunk/query-term coverage, counts, mean score when exposed), answer-relevance /
  evidence-support / unsupported-claim proxies, and the recorded Self-RAG
  verification decisions + retries + web search.
- **Grader-based ranking metrics (real metrics over grader proxies):**
  Precision@5, Recall@5, Reciprocal Rank, MAP@5, nDCG@5 from the recorded
  relevance grades (Self-RAG's own grades; Baseline RAG reuses them only on
  identical retrieval, else the optional grader). Shown per system and kept
  separate from the ground-truth block.
- **Unavailable (shown as `unavailable`, never `0`):** *ground-truth* Precision@5,
  Recall@5, MRR, MAP@5, nDCG@5 (need a relevance reference); provider token usage
  for Self-RAG when the graph does not expose it; estimated cost without
  configured pricing.
- **Require the optional evaluator (`pending`):** LLM-judged answer
  relevance/faithfulness and Baseline RAG chunk grading, until the evaluator is
  explicitly enabled and run.

### Full-screen question detail

The dashboard's main view is the question history only. Opening a row shows a
**full-screen** per-question detail that occupies the whole viewport and scrolls.
It is rebuilt from scratch on every open (one canonical rendering path), so no
content is appended twice, switching questions leaves no stale data, and opening a
question never regenerates the LLM answer or duplicates stored evaluations. Back /
Close (or `Esc`) returns to the history with the request id preserved. Each answer,
chunk list, metric, trace step and verification decision is rendered exactly once;
when both systems share the identical initial top-k, that shared retrieval's full
text is shown once (under Self-RAG) and collapsed for Baseline rather than
repeated.

### Retrieval ranking metrics (offline benchmark)

`evaluation/metrics.py` still computes Precision@5, Recall@5, MRR, MAP@5 and
nDCG@5 for the **offline** benchmark, which has a labelled ground-truth dataset
(`evaluation/dataset.json`). Binary relevance; retrieved ids are deduplicated (the
first occurrence wins) before scoring, so several chunks of one document cannot
inflate precision.

| Metric | Formula |
| --- | --- |
| Precision@5 | distinct top-5 items in the relevant set / distinct top-5 items |
| Recall@5 | distinct top-5 items in the relevant set / distinct relevant items |
| Reciprocal rank | 1 / rank of the first relevant deduped item; 0 when nothing relevant was retrieved |
| AP@5 (MAP numerator) | (sum over relevant ranks `i<=5` of Precision@i) / min(#relevant, 5) |
| nDCG@5 | `DCG@5 / IDCG@5`, binary gains, `log2(i+1)` discount: `DCG@5 = sum(rel_i / log2(i+1))`, `IDCG@5 = sum(1 / log2(i+1))` over `min(#relevant, 5)` |

The live dashboard does **not** recompute these *ground-truth* metrics (it has no
relevance reference) and shows them as unavailable with an explanation. It has a
separate grader-based block (above) computed from recorded grader grades.

### Limitations

- **Proxy, not ground truth:** the automatic lexical proxies can miss semantic
  paraphrases and can be fooled by text that merely repeats query terms; they are
  labelled as proxies and never presented as accuracy.
- **Background execution:** no durable queue (Celery/RQ) in this deployment, so
  the shadow evaluation uses in-process `BackgroundTasks`. Queued/executing tasks
  do not survive a process restart, and bounded concurrency is per process.
- **Persistence:** the SQLite database lives on the container filesystem. On an
  ephemeral free-tier host it resets on redeploy/restart. Mount a persistent disk
  (e.g. Render disk at `/app/data`) to make records survive — the same
  requirement already applies to the existing audit and LangGraph memory tables.

