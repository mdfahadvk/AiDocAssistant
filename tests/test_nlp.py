import unittest

from src.nlp.local import LocalQueryAnalyzer


class QueryAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = LocalQueryAnalyzer(
            model_name="model-not-installed-in-test-environment",
            vocabulary={
                "projects": ["WAM", "Vendor Maintenance"],
                "applications": [],
                "databases": [],
                "pages": ["History Log"],
                "apis": [],
                "stored_procedures": [],
                "technologies": [],
            },
        )

    def test_how_to_project_and_expansion(self):
        result = self.analyzer.analyze("How do I add a new employee commission in WAM?")
        self.assertEqual(result.intent, "how_to")
        self.assertEqual(result.project, "WAM")
        self.assertTrue(result.retrieval_queries)

    def test_database_intent(self):
        self.assertEqual(self.analyzer.analyze("Which database does WAM use?").intent, "database")

    def test_page_functionality_intent(self):
        result = self.analyzer.analyze("What is the History Log page used for in WAM?")
        self.assertEqual(result.intent, "page_functionality")
        self.assertEqual(result.project, "WAM")
        self.assertTrue(any("History Log" in query for query in result.retrieval_queries))

    def test_comparison_detects_multiple_projects(self):
        result = self.analyzer.analyze("Compare WAM and Vendor Maintenance")
        self.assertEqual(result.intent, "comparison")
        self.assertTrue(result.is_comparison_query)
        self.assertEqual(result.entities["project"], ["WAM", "Vendor Maintenance"])
        self.assertIsNone(result.project)

    def test_follow_up_uses_history_project(self):
        result = self.analyzer.analyze(
            "What database does it use?",
            ["What is WAM?"],
        )
        self.assertTrue(result.is_followup_query)
        self.assertEqual(result.project, "WAM")

    def test_short_section_queries_are_marked_for_local_expansion(self):
        result = self.analyzer.analyze("Introduction?")
        self.assertIn("section_lookup", result.query_types)
        self.assertIn("short_query", result.query_types)
        self.assertTrue(result.expansion_required)
        self.assertTrue(any("Introduction section" in query for query in result.retrieval_queries))

    def test_purpose_and_project_introduction_are_normalized(self):
        purpose = self.analyzer.analyze("purpose")
        self.assertIn("Purpose of the selected application or project", purpose.normalized_query)
        introduction = self.analyzer.analyze("introduce me this project")
        self.assertEqual(introduction.intent, "overview")
        self.assertIn("Introduction and purpose", introduction.normalized_query)

    def test_typo_is_corrected_before_expansion(self):
        result = self.analyzer.analyze("What are the color coding avialble?")
        self.assertEqual(result.corrected_query, "What are the color coding available?")
        self.assertTrue(any("available" in query for query in result.retrieval_queries))

    def test_specific_query_does_not_require_expansion(self):
        result = self.analyzer.analyze("How do I add a new employee commission in WAM?")
        self.assertFalse(result.expansion_required)
        self.assertEqual(len(result.retrieval_queries), 1)


if __name__ == "__main__":
    unittest.main()
