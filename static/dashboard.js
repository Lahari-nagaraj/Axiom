/* Live evaluation dashboard for Axiom.
 *
 * The dashboard shows ONLY a paginated question history (newest first). Opening a
 * row renders one full-screen question detail built from that request's persisted
 * records via /api/evaluation/requests/{id}. There is a single canonical rendering
 * path: the detail body is rebuilt from scratch on every open, so nothing is
 * appended twice and no stale data survives a switch. Each answer, chunk list,
 * metric, trace step and verification decision is rendered exactly once.
 *
 * Unavailable measurements are shown as unavailable, never as zero. No aggregate
 * summary is displayed.
 *
 * Exposes window.AxiomDashboard { init, load }.
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const state = { page: 1, pageSize: 10, currentRequestId: null };
  let historyMeta = { total_pages: 0 };
  let detailToken = 0;

  const STATUS_LABEL = {
    ok: "measured",
    unavailable: "unavailable",
    insufficient_data: "insufficient data",
    partial: "partial",
    not_evaluated: "not evaluated",
  };

  const GRADER_METRICS = [
    ["precision_at_5", "Precision@5"],
    ["recall_at_5", "Recall@5"],
    ["reciprocal_rank", "Reciprocal Rank"],
    ["map_at_5", "MAP@5"],
    ["ndcg_at_5", "nDCG@5"],
  ];

  const FORMULA_LABELS = {
    precision_at_5: "Precision@5",
    recall_at_5: "Recall@5",
    reciprocal_rank: "Reciprocal Rank",
    map_at_5: "MAP@5",
    ndcg_at_5: "nDCG@5",
  };

  const SYSTEM_LABELS = { baseline_rag: "Baseline RAG", axiom_self_rag: "Self-RAG" };

  function esc(s = "") {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
    }[c]));
  }
  function pretty(v = "") {
    return String(v === null || v === undefined ? "" : v).replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
  }
  function pct(v, digits = 1) {
    return v === null || v === undefined ? null : (Number(v) * 100).toFixed(digits) + "%";
  }
  function secs(v, digits = 2) {
    return v === null || v === undefined ? null : Number(v).toFixed(digits) + "s";
  }
  function num(v, digits = 2) {
    return v === null || v === undefined ? null : Number(v).toFixed(digits);
  }
  function fmtInt(v) {
    return v === null || v === undefined ? null : String(v);
  }
  function statusClass(status) {
    if (status === "ok") return "ok";
    if (status === "partial" || status === "insufficient_data" || status === "not_evaluated") return "warn";
    return "muted";
  }
  function statusNote(status) {
    return STATUS_LABEL[status] || status || "unavailable";
  }

  /* A metric line: shows the value when measurable, otherwise an explicit status. */
  function metricRow(label, value, status, note, display) {
    if (value === null || value === undefined) {
      const text = display || statusNote(status);
      return `<div class="cmp-row"><span>${esc(label)}</span><span class="metric-empty">${esc(text)}</span></div>`;
    }
    return `<div class="cmp-row"><span>${esc(label)}</span><span class="metric-val">${esc(value)}</span></div>`;
  }
  function badge(label, status) {
    return `<span class="badge ${statusClass(status)}">${esc(label)}</span>`;
  }
  function kv(label, value) {
    return `<div class="kv"><span>${esc(label)}</span><span>${value === null || value === undefined || value === "" ? '<span class="metric-empty">—</span>' : esc(value)}</span></div>`;
  }
  function section(title, inner) {
    return `<section class="qd-section"><h4>${esc(title)}</h4>${inner}</section>`;
  }
  function formulasBlock(formulas) {
    if (!formulas) return "";
    const items = Object.entries(formulas).map(([key, text]) =>
      `<li><strong>${esc(FORMULA_LABELS[key] || pretty(key))}:</strong> ${esc(text)}</li>`).join("");
    return `<details class="qd-details"><summary>Metric formulas &amp; assumptions</summary><ul class="formula-list">${items}</ul></details>`;
  }

  function qs(params) {
    const p = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== "" && v !== null && v !== undefined) p.append(k, v);
    });
    return p.toString();
  }

  async function fetchJSON(url) {
    const r = await fetch(url);
    let data = {};
    try { data = await r.json(); } catch (e) { data = {}; }
    if (!r.ok) throw new Error((data && data.detail) || "Request failed (" + r.status + ")");
    return data;
  }

  function setStatus(kind, message) {
    const el = $("dashStatus");
    if (!el) return;
    if (!message) { el.hidden = true; el.textContent = ""; el.className = "dash-status"; return; }
    el.hidden = false;
    el.className = "dash-status " + kind;
    el.textContent = message;
  }

  function loading(el, msg = "Loading…") { if (el) el.innerHTML = `<div class="dash-state">${esc(msg)}</div>`; }
  function empty(el, msg) { if (el) el.innerHTML = `<div class="dash-state empty">${esc(msg)}</div>`; }
  function failed(el, msg) { if (el) el.innerHTML = `<div class="dash-state error">${esc(msg)}</div>`; }

  /* ------------------------------------------------------------------ */
  /* Question history                                                    */
  /* ------------------------------------------------------------------ */
  async function loadHistory() {
    const el = $("dashHistory");
    if (!el) return;
    loading(el, "Loading history…");
    try {
      const data = await fetchJSON("/api/evaluation/history?" + qs({
        page: state.page, page_size: state.pageSize,
      }));
      historyMeta = data;
      if (!data.items || !data.items.length) {
        empty(el, "No questions recorded yet.");
        $("dashPageInfo").textContent = "—";
        $("dashPrev").disabled = true;
        $("dashNext").disabled = true;
        return;
      }
      const rows = data.items.map((it) => {
        const sr = it.self_rag, bl = it.baseline;
        const sysCell = (s) => s
          ? `<span class="badge ${s.status === "ok" ? "ok" : "err"}">${esc(s.status)}</span> <span class="metric-empty">${esc(secs(s.latency_seconds) || "")}</span>`
          : `<span class="badge muted">n/a</span>`;
        const ev = it.evaluation || {};
        return `<tr class="hist-row" data-request-id="${esc(it.request_id)}">
          <td class="mono">${esc(it.request_id.slice(0, 8))}</td>
          <td>${esc((it.created_at || "").slice(0, 19).replace("T", " "))}</td>
          <td class="q-cell">${esc(it.question)}</td>
          <td>${sysCell(bl)}</td>
          <td>${sysCell(sr)}</td>
          <td><span class="badge ${statusClass(ev.status)}">${esc(ev.label || "Not evaluated")}</span></td>
        </tr>`;
      }).join("");
      el.innerHTML = `<table class="dash-table hist"><thead><tr>
        <th>ID</th><th>Time (UTC)</th><th>Question</th><th>Baseline RAG</th><th>Self-RAG</th><th>Evaluation</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
      el.querySelectorAll(".hist-row").forEach((row) =>
        row.addEventListener("click", () => openDetail(row.dataset.requestId)));
      $("dashPageInfo").textContent = `Page ${data.page} / ${Math.max(1, data.total_pages)} · ${data.total} questions`;
      $("dashPrev").disabled = !data.has_prev;
      $("dashNext").disabled = !data.has_next;
    } catch (e) {
      failed(el, e.message);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Detail renderers (one canonical path; each item rendered once)      */
  /* ------------------------------------------------------------------ */
  function tokenText(tok) {
    if (!tok || !tok.available) return null;
    const total = fmtInt(tok.total_tokens) || "—";
    return `${total} tokens${tok.estimated_cost_usd != null ? " · $" + num(tok.estimated_cost_usd, 6) : " · cost unavailable"}`;
  }

  function chunkList(title, chunks, note, opts) {
    const condense = !!(opts && opts.condense);
    let inner;
    if (!chunks || !chunks.length) {
      inner = `<div class="metric-empty">none recorded</div>`;
    } else {
      const items = chunks.map((c) => {
        const text = c.text || "";
        const shown = text.length > 600 ? text.slice(0, 600) + "…" : text;
        let body;
        if (condense) {
          body = `<div class="metric-empty">${esc((opts && opts.condenseNote) || "full chunk text not repeated here — shown once under Self-RAG's initial retrieval")}</div>`;
        } else if (text) {
          body = `<div class="chunk-text">${esc(shown)}</div>`;
        } else {
          body = `<div class="metric-empty">no chunk text recorded</div>`;
        }
        return `<li class="chunk-item">
          <div class="chunk-head">
            <span class="badge muted">#${esc(c.rank)}</span>
            <span class="chunk-src mono">${esc(c.source || c.title || "(no source)")}</span>
            ${c.type === "web" ? '<span class="badge warn">web</span>' : ""}
            ${c.page ? `<span class="metric-empty">p.${esc(c.page)}</span>` : ""}
            ${c.score != null ? `<span class="metric-empty">score ${esc(num(c.score, 4))}</span>` : '<span class="metric-empty">score n/a</span>'}
          </div>
          ${body}
        </li>`;
      }).join("");
      inner = `<ul class="chunk-list">${items}</ul>`;
    }
    return `<div class="qd-block"><h5>${esc(title)}</h5>${note ? `<div class="dash-note">${esc(note)}</div>` : ""}${inner}</div>`;
  }

  function sameChunks(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      const x = a[i], y = b[i];
      const xid = x.chunk_id || (x.type || "") + "|" + (x.source || "") + "|" + (x.text || "");
      const yid = y.chunk_id || (y.type || "") + "|" + (y.source || "") + "|" + (y.text || "");
      if (xid !== yid) return false;
    }
    return true;
  }

  function autoRetrievalRows(b) {
    if (!b) return '<div class="metric-empty">not computed</div>';
    const v = (x, d) => (x === null || x === undefined ? null : num(x, d));
    const rows = [
      metricRow("Chunks retrieved", fmtInt(b.chunks_retrieved), b.status),
      metricRow("Chunks with text", fmtInt(b.chunks_with_text), b.status),
      metricRow("Distinct sources", fmtInt(b.distinct_sources), b.status),
      metricRow("Mean query-term coverage", v(b.mean_query_term_coverage, 3), b.status, statusNote(b.status)),
      metricRow("Top-rank query-term coverage", v(b.top_rank_query_term_coverage, 3), b.status, statusNote(b.status)),
      metricRow("Query terms covered (any chunk)", v(b.query_terms_covered_by_any_chunk, 3), b.status, statusNote(b.status)),
      metricRow("Mean retrieval score", v(b.mean_retrieval_score, 4), b.status, null,
        b.mean_retrieval_score === null || b.mean_retrieval_score === undefined ? "retriever exposed no score" : null),
    ];
    return rows.join("") + (b.limitations ? `<div class="dash-note">${esc(b.limitations)}</div>` : "");
  }

  function answerQualityRows(b) {
    if (!b) return '<div class="metric-empty">not computed</div>';
    const v = (x, d) => (x === null || x === undefined ? null : num(x, d));
    const rows = [
      metricRow("Answer relevance", v(b.answer_relevance, 3), b.status, statusNote(b.status)),
      metricRow("Evidence support", v(b.evidence_support, 3), b.status, statusNote(b.status)),
      metricRow("Unsupported claims", fmtInt(b.unsupported_claims), b.status, statusNote(b.status)),
      metricRow("Claims total", fmtInt(b.claims_total), b.status, statusNote(b.status)),
      metricRow("Unsupported terms", fmtInt(b.unsupported_terms_total), b.status, statusNote(b.status)),
    ];
    let html = rows.join("");
    if (b.unsupported_terms && b.unsupported_terms.length) {
      html += `<div class="dash-note">Unsupported terms: ${esc(b.unsupported_terms.join(", "))}</div>`;
    }
    if (b.limitations) html += `<div class="dash-note">${esc(b.limitations)}</div>`;
    return html;
  }

  function graderMetricRows(b) {
    if (!b) return '<div class="metric-empty">not computed</div>';
    const ms = b.metric_status || {};
    const rows = GRADER_METRICS.map(([key, label]) =>
      metricRow(label, num(b[key], 4), ms[key] || b.status, b.reason, b[key] == null ? statusNote(ms[key] || b.status) : null)
    ).join("");
    const scope = [
      kv("Ranked chunks", fmtInt(b.ranked_total)),
      kv("Judged chunks", fmtInt(b.graded_total)),
      kv("Ungraded chunks", fmtInt(b.ungraded_total)),
      kv("Judged-relevant chunks", fmtInt(b.relevant_total_judged)),
      kv("Judgment complete", b.complete_judgment ? "yes" : "no"),
    ].join("");
    let html = `<div class="qd-block"><h5>Grader-based ranking metrics ${badge(pretty(b.source || "grader"), b.status)}</h5>
      <div class="qd-kv">${scope}</div>${rows}`;
    if (b.reason) html += `<div class="dash-note">Reason: ${esc(b.reason)}</div>`;
    if (b.reuse_note) html += `<div class="dash-note">${esc(b.reuse_note)}</div>`;
    if (b.message) html += `<div class="dash-note">${esc(b.message)}</div>`;
    if (b.gain) html += `<div class="dash-note">Gain function: ${esc(b.gain)}</div>`;
    html += formulasBlock(b.formulas);
    if (b.limitations) html += `<div class="dash-note">${esc(b.limitations)}</div>`;
    return html + `</div>`;
  }

  function traditionalBlock(block) {
    const names = [["precision_at_5", "Precision@5"], ["recall_at_5", "Recall@5"], ["mrr", "MRR"], ["map_at_5", "MAP@5"], ["ndcg_at_5", "nDCG@5"]];
    const rows = names.map(([, label]) => metricRow(label, null, "unavailable", null, "unavailable")).join("");
    return rows + (block && block.message ? `<div class="dash-note">${esc(block.message)}</div>` : "");
  }

  function gradesList(grades) {
    if (!grades || !grades.length) {
      return `<div class="qd-block"><h5>Self-RAG relevance grades</h5><div class="metric-empty">no grades recorded</div></div>`;
    }
    const items = grades.map((g) => {
      const rank = (g.rank === null || g.rank === undefined) ? "?" : g.rank;
      const round = (g.retrieval_round === null || g.retrieval_round === undefined) ? "" : ` <span class="metric-empty">round ${esc(g.retrieval_round)}</span>`;
      return `<li><span class="badge muted">#${esc(rank)}</span> ${esc(g.source || g.title || "(no source)")}${round} <span class="badge ${g.relevant ? "ok" : "muted"}">${g.relevant ? "relevant" : "not relevant"}</span></li>`;
    }).join("");
    return `<div class="qd-block"><h5>Self-RAG relevance grades</h5>
      <div class="dash-note">The workflow's own per-chunk relevance grades (a self-assessment, not ground truth).</div>
      <ul class="src-list">${items}</ul></div>`;
  }

  function systemColumn(key, sys, opts) {
    opts = opts || {};
    const label = SYSTEM_LABELS[key] || key;
    if (!sys) {
      return `<div class="qd-col"><div class="qd-col-head">${esc(label)}</div><div class="metric-empty">not evaluated for this question</div></div>`;
    }
    const isSelfRag = key === "axiom_self_rag";
    const ver = sys.verification || {};
    const verRows = isSelfRag
      ? [
          kv("IsSUP (reported)", ver.support_status ? pretty(ver.support_status) : ""),
          kv("IsUSE (reported)", ver.usefulness ? pretty(ver.usefulness) : ""),
        ]
      : [];
    const runStats = [
      kv("Status", sys.status),
      kv("View", pretty(sys.view)),
      kv("Latency", secs(sys.latency_seconds)),
      kv("Route", sys.route),
      kv("Retries", fmtInt(sys.retries)),
      kv("Web search", sys.used_web_search ? "yes" : "no"),
      ...verRows,
      kv("Tokens", tokenText(sys.tokens)),
    ].join("");
    const verNote = isSelfRag
      ? (ver.note || "")
      : "Baseline RAG is a single-pass retrieval baseline; it has no IsSUP/IsUSE verification step.";

    const runDetails = `<div class="qd-block"><h5>Run details</h5><div class="qd-kv">${runStats}</div>${verNote ? `<div class="dash-note">${esc(verNote)}</div>` : ""}</div>`;

    const answer = `<div class="qd-block"><h5>Answer</h5>
      <div class="drawer-answer">${esc(sys.answer || "") || '<span class="metric-empty">no answer recorded</span>'}</div></div>`;

    let evidence = "";
    if (isSelfRag) {
      const finalSame = sameChunks(sys.retrieved_chunks, sys.final_evidence);
      evidence += chunkList("Initial retrieval (chunks)", sys.retrieved_chunks,
        "Self-RAG's first-pass retrieval, before grading, rewriting or web search.");
      evidence += chunkList("Final evidence used to answer", sys.final_evidence,
        finalSame
          ? "Every initially retrieved chunk was judged relevant and kept, so this equals the initial retrieval above; the full text is not repeated."
          : "The post-grading evidence Self-RAG actually used — distinct from the initial retrieval above.",
        { condense: finalSame, condenseNote: "identical to the initial retrieval above — full chunk text not repeated" });
      evidence += gradesList(sys.grades);
    } else {
      evidence += chunkList(
        "Retrieved chunks (answer context)",
        sys.retrieved_chunks,
        opts.sharedRetrieval
          ? "Baseline RAG answers directly from these chunks. They are the shared initial top-k, identical to Self-RAG's initial retrieval, so the full text is shown once under Self-RAG."
          : "Baseline RAG answers directly from these chunks; its evidence is its retrieval, shown once.",
        { condense: !!opts.sharedRetrieval }
      );
    }

    return `<div class="qd-col">
      <div class="qd-col-head">${esc(label)} ${badge(sys.status, sys.status === "ok" ? "ok" : "err")}${sys.used_web_search ? ' <span class="badge warn">web</span>' : ""}</div>
      ${sys.error ? `<div class="drawer-error"><strong>Failure:</strong> ${esc(sys.error)} <span class="badge err">${esc(sys.error_category || "unknown")}</span></div>` : ""}
      ${runDetails}
      ${answer}
      ${evidence}
      <div class="qd-block"><h5>Retrieval-quality assessment (lexical proxy)</h5>${autoRetrievalRows(sys.automatic)}</div>
      <div class="qd-block"><h5>Answer relevance, faithfulness &amp; evidence coverage (proxies)</h5>${answerQualityRows(sys.answer_quality)}</div>
    </div>`;
  }

  function renderDetail(d) {
    const systems = d.systems || {};
    const baseline = systems.baseline_rag;
    const selfRag = systems.axiom_self_rag;

    const overview = [
      kv("Request ID", d.request_id),
      kv("Created (UTC)", (d.created_at || "").slice(0, 19).replace("T", " ")),
      kv("Execution status", d.status),
      kv("Route", d.route),
      kv("Latency", secs(d.latency_seconds)),
      kv("Retries", fmtInt(d.retries)),
      kv("Web search", d.used_web_search ? "yes" : "no"),
      kv("Tokens", tokenText(d.tokens)),
      kv("Retrieval recorded", d.retrieval_available ? "yes" : "no"),
      kv("Thread", d.thread_id),
    ].join("");

    const qualityBlock = d.quality_status === "ok"
      ? (d.quality_scores || []).map((q) => kv(pretty(q.method), "relevance " + num(q.answer_relevance, 2) + " · faithfulness " + num(q.faithfulness, 2))).join("")
      : `<div class="metric-empty">${esc(statusNote(d.quality_status))} — the optional evaluator has not stored a score</div>`;

    const trace = (d.trace || []).length
      ? section("Self-RAG workflow trace", `<ol class="trace-list">${d.trace.map((t) => `<li>${esc(t)}</li>`).join("")}</ol>`)
      : "";

    const cited = (!selfRag || !(selfRag.final_evidence || []).length) && (d.sources || []).length
      ? section("Cited sources", `<ul class="src-list">${d.sources.map((s) => `<li>${esc(s.title || s.source || "Evidence")}${s.url ? ` · <a href="${esc(s.url)}" target="_blank" rel="noopener">open ↗</a>` : ""}</li>`).join("")}</ul>`)
      : "";

    const graderNote = `<div class="dash-note">Grader-based ranking metrics are computed from an automatic relevance grader (Self-RAG's recorded per-chunk grades, or the optional evaluator for Baseline RAG). They are a proxy, not ground-truth accuracy, and are kept separate from the conventional ground-truth metrics below.</div>`;

    const shared = !!d.shared_initial_retrieval;

    return `
      ${d.error ? `<div class="drawer-error"><strong>Failure:</strong> ${esc(d.error)} <span class="badge err">${esc(d.error_category || "unknown")}</span></div>` : ""}
      ${section("Request overview", `<div class="qd-kv qd-kv-wide">${overview}</div>`)}
      ${section("Question", `<div class="drawer-q">${esc(d.question)}</div>`)}
      <section class="qd-section">
        <h4>Per-system evaluation</h4>
        ${graderNote}
        ${shared ? '<div class="dash-note">Both systems retrieved the identical initial top-k, so that shared retrieval is shown in full once (under Self-RAG) and not repeated.</div>' : ""}
        <div class="cmp-grid qd-systems">
          ${systemColumn("baseline_rag", baseline, { sharedRetrieval: shared })}
          ${systemColumn("axiom_self_rag", selfRag, { sharedRetrieval: shared })}
        </div>
      </section>
      ${section("Self-RAG Grader-Based Ranking Metrics", graderMetricRows(selfRag && selfRag.grader_ranking))}
      ${section("Baseline RAG Grader-Based Ranking Metrics", graderMetricRows(baseline && baseline.grader_ranking))}
      ${section("Conventional ground-truth retrieval metrics", `${traditionalBlock(d.traditional_metrics)}<div class="dash-note">These apply to both systems and require human relevance judgments or a labelled ground truth, which the live system does not have automatically.</div>`)}
      ${trace}
      ${cited}
      ${section("Optional evaluator scores", qualityBlock)}
      ${section("Methodology & limitations", `<div class="dash-note">${esc(d.method_note || "")} <span class="mono">(${esc(d.method || "")})</span></div>${d.grader_limitations ? `<div class="dash-note">${esc(d.grader_limitations)}</div>` : ""}`)}
    `;
  }

  /* ------------------------------------------------------------------ */
  /* Full-screen question detail                                         */
  /* ------------------------------------------------------------------ */
  async function openDetail(requestId) {
    const panel = $("questionDetail");
    const body = $("qdBody");
    if (!panel || !body) return;
    state.currentRequestId = requestId;
    detailToken += 1;
    const token = detailToken;
    panel.hidden = false;
    if ($("qdTitle")) $("qdTitle").textContent = "Question " + requestId.slice(0, 8);
    if ($("qdMeta")) $("qdMeta").textContent = "";
    loading(body, "Loading question evaluation…");
    let d;
    try {
      d = await fetchJSON("/api/evaluation/requests/" + encodeURIComponent(requestId));
    } catch (e) {
      if (token === detailToken) failed(body, e.message);
      return;
    }
    if (token !== detailToken) return; // a newer question was opened; drop stale data
    if ($("qdMeta")) $("qdMeta").textContent = (d.created_at || "").slice(0, 19).replace("T", " ") + " · " + (d.status || "");
    body.innerHTML = renderDetail(d);
    body.scrollTop = 0;
  }

  function closeDetail() {
    const panel = $("questionDetail");
    const body = $("qdBody");
    detailToken += 1; // invalidate any in-flight fetch
    state.currentRequestId = null;
    if (body) body.innerHTML = "";
    if (panel) panel.hidden = true;
  }

  /* ------------------------------------------------------------------ */
  /* Init + public API                                                   */
  /* ------------------------------------------------------------------ */
  async function load() {
    setStatus("", "");
    loading($("dashHistory"), "Loading history…");
    try {
      await loadHistory();
      if (((historyMeta || {}).total || 0) === 0) {
        setStatus("info", "No questions recorded yet. Ask a question in the Incident Copilot to start collecting evaluation data.");
      }
    } catch (e) {
      setStatus("error", "Could not load the question history: " + e.message);
      failed($("dashHistory"), e.message);
    }
  }

  function init() {
    const refresh = $("dashRefresh");
    if (refresh) refresh.addEventListener("click", () => { state.page = 1; load(); });
    const prev = $("dashPrev"), next = $("dashNext");
    if (prev) prev.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; loadHistory(); } });
    if (next) next.addEventListener("click", () => { if (state.page < (historyMeta.total_pages || 1)) { state.page += 1; loadHistory(); } });
    const back = $("qdBack"), close = $("qdClose");
    if (back) back.addEventListener("click", closeDetail);
    if (close) close.addEventListener("click", closeDetail);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });
  }

  window.AxiomDashboard = { init, load };
})();
