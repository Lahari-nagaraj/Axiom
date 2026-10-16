# Axiom Evaluation

A minimal, credible harness that compares two systems on the **same** questions:

| System | What runs | Self-RAG features |
| --- | --- | --- |
| `baseline_rag` | raw top-5 retrieval → one Groq generation → answer | none (no graph, no grading, no retries, no web fallback) |
| `axiom_self_rag` | the existing LangGraph Self-RAG workflow (`src.self_rag.run_self_rag`) | unchanged |

The baseline deliberately does **not** import `src.self_rag`. It reuses only the
configured retriever (`src.vectorstore.get_retriever`), the configured Groq model,
and settings, so the comparison is fair. No application or frontend code is
modified.

This README explains every metric in plain language so the results can be
presented to a project guide. Nothing here claims one system is better — read the
actual numbers.

---

## What is measured

**Automatic** (computed from the run, no human input):

- **Retrieval Precision@5** and **Retrieval Recall@5** over the shared raw top-5
  retrieval.
- **Latency P50 / P95**, reported separately for successful and failed requests.
- **Success Rate** and **Failure Rate**, plus a **Failure Categories** breakdown.
- **Token Usage**, only when the provider actually returns it.

**Manual** (human-scored, blank until reviewed):

- **Answer Correctness (manual, 1–5)** and **Faithfulness (manual, 1–5)**.

**Not yet available** (explicitly reported as `null`, never `0`):

- **Citation Precision** and **Citation Recall** (see below for the annotations
  required).

Every metric is labelled `automatic` or `manual` in the outputs, and the full
formula list is embedded in `summary_<RUN_ID>.json` under `metric_definitions`.

---

## Metric reference

### Retrieval Precision@5 (automatic)

- **What it measures:** how on-topic the retrieved runbooks are.
- **Formula:**
  `(number of distinct retrieved documents in the top-5 that are in the ground truth) / (number of distinct retrieved documents in the top-5)`
  The denominator is at most 5.
- **Deduplication:** several chunks from the same document count once, so
  duplicate chunks cannot inflate precision.
- **Scope:** the shared *raw* top-5 retrieval, which is identical for both
  systems. It does **not** score the documents a system ultimately used after
  Self-RAG grading, query rewriting or web search (those live in `used_sources`).
- **Worked example:** ground truth `{checkout-api-runbook.md}`; the top-5 chunks
  come from `checkout-api-runbook.md ×3` and `payments-high-cpu-runbook.md ×2`.
  Deduplicated, that is `{checkout, payments}` → 1 of 2 relevant → **0.5**.
  (Without deduplication it would have been 3/5 = 0.6, which is why we dedupe.)
- **High / low:** high means the retriever mostly returns relevant runbooks; low
  means it returns unrelated documents.
- **Excluded:** questions with no ground truth (ambiguous/unanswerable) and rows
  where no raw retrieval was recorded. Rows with zero retrieved documents have no
  denominator, so precision is reported as unavailable (`null`), not `0`.
- **Plain English:** "When Axiom searches the runbooks, how often is what it finds
  actually relevant?"

### Retrieval Recall@5 (automatic)

- **What it measures:** whether the needed runbook(s) appear at all in the top-5.
- **Formula:**
  `(number of distinct ground-truth documents present in the top-5) / (total number of distinct ground-truth documents)`
- **Worked example:** ground truth `{checkout-api-runbook.md, deployment-rollback-sop.md}`;
  the top-5 contain `checkout` and `payments` → 1 of 2 required documents found →
  **0.5**.
- **High / low:** high means the required runbook is usually found; low means it is
  usually missed (a retriever or knowledge-base coverage problem).
- **Excluded:** questions with no ground-truth sources (reported as
  `retrieval_excluded_no_ground_truth`). When retrieval ran but returned nothing,
  recall is a genuine **0.0** (a real miss) and the row is counted
  (`retrieval_empty_retrieval`); precision is `null` because there is no
  denominator.
- **Plain English:** "Does the search actually surface the runbook we needed?"

> Retrieval quality and answer quality are different things. A system can retrieve
> perfectly and still answer poorly, or vice versa. Keep Precision@5/Recall@5
> (retrieval) separate from Answer Correctness/Faithfulness (answers).

### Answer Correctness (manual, 1–5)

- **What it measures:** whether the answer is factually right, judged against the
  question's `expected_answer_points`.
- **Rubric:** 1 = incorrect, 2 = mostly incorrect, 3 = partially correct,
  4 = mostly correct, 5 = fully correct.
- **Formula:** `mean of the reviewed scores`; `n` = number of reviewed answers.
- **Worked example:** reviewed scores `[5, 3, 4]` → mean **4.0**, n = 3.
- **High / low:** high means answers match the expected points; low means they are
  wrong or irrelevant.
- **Excluded:** failed requests (they have no answer to judge) and out-of-range
  values (counted in `manual_out_of_range_values`, not averaged). Blank cells mean
  "not reviewed" and are never treated as 0.
- **Plain English:** "How accurate were the answers?"

### Faithfulness (manual, 1–5)

- **What it measures:** whether the answer's claims are grounded in the evidence
  the system actually retrieved/cited.
- **Rubric:** 1 = unsupported or contradicts the evidence, 3 = mixed,
  5 = fully grounded in cited sources.
- **Formula:** `mean of the reviewed scores`; `n` = number of reviewed answers.
- **Worked example:** reviewed scores `[4, 5]` → mean **4.5**, n = 2.
- **High / low:** high means claims are backed by evidence; low means
  unsupported claims or hallucination.
- **Excluded:** failed requests; blanks are never treated as 0.
- **Plain English:** "Can we trust that the answer is backed by the runbooks?"

### Citation Precision and Citation Recall (not yet available)

These are **defined but not computed**, because the results contain no per-claim
citation annotations, and we will not invent them or use an LLM judge.

- **Citation Precision** = `supported cited claims / all cited claims`
  (of the claims the answer cites, how many are actually supported by the cited
  evidence).
- **Citation Recall** = `required evidence items covered by citations / all required evidence items`
  (of the evidence needed to answer, how much is covered by citations).

- **Rule:** if a denominator is 0, the metric is `null` (unavailable), never `0`.
- **Required annotations to enable them:**
  1. Per answer: the number of cited claims and how many are supported by the cited evidence.
  2. Per question: the list of required evidence items and which are covered by citations.
- **Worked example (illustrative, not yet computed):** an answer cites 4 claims,
  3 supported → Citation Precision = 3/4 = **0.75**; 2 of 3 required evidence items
  are cited → Citation Recall = 2/3 ≈ **0.67**.
- **Plain English:** "Do the citations actually back up what the answer says, and
  do they cover everything the answer relies on?" (To be reported once reviewed.)

### Latency P50 / P95 (automatic)

- **What it measures:** response time per question (wall time including retries).
- **Formula:** P50 = median response time; P95 = linear-interpolation percentile,
  `rank = (n − 1) × p/100`.
- **Worked example:** successful latencies `[10s, 20s, 30s]` → P50 = **20s**,
  P95 = **29s**.
- **Separation:** successful and failed requests are reported in **separate**
  blocks (`latency_seconds.successful` and `latency_seconds.failed`). Fast quota
  failures can never make successful responses look faster.
- **High / low:** lower is faster; P95 shows worst-case behaviour the mean hides.
- **Excluded:** none from their own block; missing latencies count as no sample
  (`n` excludes them).
- **Plain English:** "How fast are successful answers, and how bad are the slow
  ones?"

### Success Rate / Failure Rate (automatic)

- **What they measure:** request reliability.
- **Formula:** `success rate = successful requests / total requests`;
  `failure rate = failed requests / total requests`.
- **Worked example:** 30 questions, 26 succeeded → Success Rate **0.8667**,
  Failure Rate **0.1333**.
- **Note:** a *successful* request only means the pipeline returned without
  raising — it does **not** mean the answer was correct. Success rate ≠ accuracy.
- **Plain English:** "How often did the system finish without error?"

### Failure Categories (automatic)

- **What they measure:** why requests failed, separated by type.
- **Formula:** count of failed requests per `error_category`.
- **Categories:** `quota_or_rate_limit` (provider 429/TPM/TPD), `parser`
  (structured-output/validation errors), `auth`, `network`, `application`, and
  `unknown` (older files without the column).
- **Why it matters:** provider quota failures are kept distinct from application
  defects. A quota failure is not a bug in Axiom; a `parser` failure is.
- **Plain English:** "Were failures due to rate limits, parsing, or actual bugs?"

### Token Usage (automatic, when available)

- **What it measures:** provider-reported token consumption (prompt, completion,
  total), summed over rows that returned numbers.
- **Rule:** if the provider returns no usage, values are `null`/blank — never
  invented. `run_self_rag` does not expose usage, so `axiom_self_rag` rows are
  always blank; the baseline records usage only when Groq returns it.
- **Plain English:** "How many tokens did the run consume, where the provider told
  us?"

**Estimated cost is not implemented.** No pricing is assumed; if the dashboard
needs it, add explicit pricing constants and label the result an estimate.

---

## Prerequisites

- Populate `.env` (see `.env.example`) with `GROQ_API_KEY` and `PINECONE_API_KEY`
  (and `TAVILY_API_KEY` if you want the Axiom web fallback exercised). The
  harness never prints these values and strips them from error messages.
- The `documents/` runbooks must already be ingested into the Pinecone index and
  namespace that `.env` points at: `python data_ingestion.py`.

## Exact commands (Windows PowerShell)

```powershell
# From the project root
cd D:\nipuna-btech\projects\Axiom-main

# 1. Install dependencies (no new packages are required for the evaluation)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (optional) make sure the knowledge base is loaded
python data_ingestion.py

# 2. Validate the dataset only (no API calls)
python -m evaluation.run_evaluation --validate-only

# 3. Run both systems on all 30 questions
python -m evaluation.run_evaluation
#    or reuse a run id to keep existing manual scores
python -m evaluation.run_evaluation --run-id 20261016T122620Z

#    Smoke test on 3 questions first if you prefer
python -m evaluation.run_evaluation --limit 3

# 4. Generate the summary (no API calls)
python evaluation\metrics.py
#    or point at a specific run
python evaluation\metrics.py --results evaluation\results\per_question_<RUN_ID>.csv
```

> If `python -m evaluation...` is blocked by your execution policy, run
> `python evaluation\run_evaluation.py` instead — the script adds the project
> root to `sys.path` itself.

## Manually scoring results

Open the newest `evaluation\results\per_question_<RUN_ID>.csv` in a spreadsheet or
editor and fill in, for each row:

| Column | Meaning |
| --- | --- |
| `answer_correctness_manual` | 1 = incorrect, 3 = partially correct, 5 = fully correct |
| `faithfulness_manual` | 1 = unsupported/contradicts evidence, 3 = mixed, 5 = fully grounded in cited sources |
| `manual_notes` | free text justification |

Leave a cell blank if you cannot judge a row. `metrics.py` only averages the cells
you actually filled in and reports the count (`n`).

### Scores are preserved

Manual scores are stored durably in `evaluation\results\manual_scores.csv`, keyed
by `(run_id, system_version, question_id)`:

- A regenerated `per_question_<RUN_ID>.csv` with blank manual columns will have
  your reviewed scores merged back in automatically.
- Scores from other run ids are never overwritten or dropped.
- Re-run with `--run-id <same id>` to rebuild the same run while keeping scores; a
  new run id starts a fresh score set, and the old scores remain in the store.

## Offline reliability tests

These tests mock all LLM calls and make no network requests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

They cover retrieval precision/recall (duplicate sources, empty retrieval, missing
ground truth, missing raw retrieval), successful-vs-failed latency and P50/P95,
success/failure rates and error categories, missing vs reviewed manual scores,
manual-score preservation, latest-run selection, summary CSV/JSON consistency, and
the "unavailable is null, not zero" rule.

## Output files

All outputs live under `evaluation\results\`:

| File | Contents |
| --- | --- |
| `per_question_<RUN_ID>.csv` | one row per question × system: answer, raw `retrieved_sources`, `used_sources`, latency, retries, `error`/`error_category`, usage, blank manual columns |
| `per_question_<RUN_ID>.json` | identical data as JSON |
| `per_question_latest.csv` | convenience copy of the most recent run (never selected as a canonical run by `metrics.py`) |
| `manual_scores.csv` | durable manual scores keyed by run/system/question |
| `summary_<RUN_ID>.csv` | one row per system with aggregate metrics (sample counts included) |
| `summary_<RUN_ID>.json` | full summary: `metric_definitions`, per-system `automatic` and `manual` blocks, exclusions |

## Dataset

`evaluation\dataset.json` contains 30 unique incident-troubleshooting questions
grounded in three runbooks, spanning five categories:

- `direct` — a single runbook step or fact.
- `paraphrased` — same ground truth, different wording.
- `multi_document` — requires more than one runbook.
- `ambiguous` — no service/symptom named; correct behaviour is to ask.
- `unanswerable` — not present in any runbook; correct behaviour is to say so.

Each entry has `id`, `question`, `category`, `relevant_sources` (ground-truth
filenames), `expected_answer_points`, and `answerable`. `--validate-only` checks
all of this, including that no two questions are duplicates and that every
answerable question lists at least one real source file. Questions with no
ground-truth sources (the ambiguous/unanswerable ones) are excluded from the
retrieval metrics and reported as `retrieval_excluded_no_ground_truth`.

## Checks you should perform

1. `python -m evaluation.run_evaluation --validate-only` prints `Dataset OK`.
2. A full run completes without crashing; inspect `success_rate`, `failure_rate`
   and `failure_categories` in `summary_<RUN_ID>.json` (failures are reported, not
   hidden).
3. Confirm `baseline_rag` rows show `route = internal_retrieval` and `retries = 0`
   (proving no Self-RAG features leaked into the baseline).
4. Confirm Axiom rows use a unique thread id per question (`eval-<run_id>-<qid>`)
   and that `used_sources` reflects what Axiom actually used.
5. Spot-check `automatic.retrieval` in the summary against a few rows by hand.
6. Only after you have filled the manual columns, confirm `manual.status =
   "scored"` and that `n` equals the number of rows you scored.
7. Verify token usage appears for the baseline only if Groq returned it, and is
   blank/`available: false` for Axiom (which does not expose usage).
8. Verify Citation Precision/Recall show `not_yet_available` until per-claim
   annotations exist.

## Limitations (reported honestly)

- **Citation Precision / Recall** require annotations the results do not yet
  contain; they are `null`/`not_yet_available`, not zero.
- **Axiom token usage** is not exposed by `run_self_rag`, so it is always blank.
- **Estimated cost** is not implemented.
- **Success rate is not accuracy.** With provider quota failures, the successful
  subset can be biased; always read it alongside Answer Correctness and the
  failure categories.
- Historical result files produced before the current runner lack
  `retrieval_available`, `used_sources` and `error_category`; `metrics.py` reads
  them without crashing (missing retrieval is treated as available, and their
  failures bucket as `unknown`), but regenerate a run for fully correct metrics.
