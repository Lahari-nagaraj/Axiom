"""Static checks for the question-history dashboard and the full-screen detail.

These read the shipped template and dashboard script so a regression that
re-introduces aggregate panels, manual-label UI, the narrow drawer, or duplicated
detail rendering is caught without a browser or a network.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "index.html"
DASHBOARD_JS = ROOT / "static" / "dashboard.js"
STYLES = ROOT / "static" / "styles.css"


class DashboardTemplateTests(unittest.TestCase):
    def setUp(self):
        self.html = TEMPLATE.read_text(encoding="utf-8")

    def test_history_panel_present(self):
        for marker in ("dashHistory", "dashPrev", "dashNext", "dashRefresh", "dashStatus"):
            self.assertIn(f'id="{marker}"', self.html, marker)

    def test_aggregate_panels_removed(self):
        for removed in ("dashCards", "dashComparison", "dashWorkflow", "dashDaily", "dashQuality"):
            self.assertNotIn(f'id="{removed}"', self.html, removed)

    def test_full_screen_detail_replaces_the_drawer(self):
        for marker in ("questionDetail", "qdBody", "qdBack", "qdClose", "qdTitle"):
            self.assertIn(f'id="{marker}"', self.html, marker)
        # The old narrow right-side drawer is gone.
        for removed in ("traceDrawer", "drawerBody", "drawerTitle", "drawer-panel"):
            self.assertNotIn(removed, self.html, removed)
        self.assertIn("Back to Question History", self.html)


class DashboardScriptTests(unittest.TestCase):
    def setUp(self):
        self.js = DASHBOARD_JS.read_text(encoding="utf-8")

    def test_loads_history_and_detail_only(self):
        self.assertIn("/api/evaluation/history", self.js)
        self.assertIn("/api/evaluation/requests/", self.js)

    def test_no_aggregate_endpoint_calls(self):
        for removed in ("/api/evaluation/summary", "/api/evaluation/comparison", "/api/evaluation/daily"):
            self.assertNotIn(removed, self.js, removed)

    def test_no_manual_labeling_ui(self):
        self.assertNotIn("/labels", self.js)
        self.assertNotIn("Relevant</button>", self.js)

    def test_single_rendering_path_and_stale_guard(self):
        # One detail renderer, one call site, and an explicit stale-response guard.
        self.assertEqual(self.js.count("function renderDetail("), 1)
        self.assertEqual(self.js.count("= renderDetail(d)"), 1)
        self.assertIn("body.innerHTML = renderDetail(d)", self.js)
        self.assertIn("detailToken", self.js)
        self.assertIn("state.currentRequestId", self.js)
        # The detail is fully rebuilt, never appended to.
        for forbidden in ("insertAdjacentHTML", "appendChild(document.createElement", "drawerBody", "traceDrawer"):
            self.assertNotIn(forbidden, self.js, forbidden)

    def test_each_system_and_metric_rendered_once(self):
        # A single canonical answer renderer used by both system columns.
        self.assertEqual(self.js.count("<h5>Answer</h5>"), 1)
        # Each chunk list heading appears exactly once.
        for heading in ("Initial retrieval (chunks)", "Final evidence used to answer", "Retrieved chunks (answer context)"):
            self.assertEqual(self.js.count(heading), 1, heading)
        # Grader metric sections appear once per system.
        self.assertEqual(self.js.count("Self-RAG Grader-Based Ranking Metrics"), 1)
        self.assertEqual(self.js.count("Baseline RAG Grader-Based Ranking Metrics"), 1)
        # Conventional ground-truth metrics are rendered only once.
        self.assertEqual(self.js.count("Conventional ground-truth retrieval metrics"), 1)

    def test_shared_retrieval_text_is_not_repeated(self):
        # When both systems share the initial top-k, the full chunk text is
        # collapsed for Baseline rather than repeated.
        self.assertIn("shared_initial_retrieval", self.js)
        self.assertIn("condense", self.js)
        self.assertIn("full chunk text not repeated here", self.js)
        # Self-RAG's final evidence is collapsed too when it equals the initial
        # retrieval, so identical full text never appears twice.
        self.assertIn("sameChunks", self.js)

    def test_close_clears_detail_body(self):
        self.assertIn("function closeDetail(", self.js)
        self.assertIn("body.innerHTML = \"\"", self.js)


class DashboardStyleTests(unittest.TestCase):
    def test_full_screen_css(self):
        css = STYLES.read_text(encoding="utf-8")
        self.assertIn(".question-detail{position:fixed;inset:0", css)
        self.assertIn("overflow:auto", css)


class MetricsModuleTests(unittest.TestCase):
    def test_live_metrics_has_no_labeling_or_dataset_dependency(self):
        source = (ROOT / "src" / "live_metrics.py").read_text(encoding="utf-8")
        for removed in (
            "build_relevance_labels",
            "save_relevance_label",
            "delete_relevance_label",
            "dataset.json",
            "live_relevance_labels",
        ):
            self.assertNotIn(removed, source, removed)

    def test_no_hard_coded_live_questions_or_fabricated_scores(self):
        source = (ROOT / "src" / "auto_eval.py").read_text(encoding="utf-8")
        # The evaluator is deterministic: no benchmark dataset and no literal
        # relevance labels are baked into the live path.
        self.assertNotIn("dataset.json", source)
        self.assertNotIn("ground_truth", source)


if __name__ == "__main__":
    unittest.main()
