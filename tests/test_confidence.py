import unittest

from src.evaluation.confidence import AnswerConfidenceService, ConfidenceWeights


def source(source_id, text, document="WAM.pdf", project="WAM"):
    return {
        "id": source_id,
        "text": text,
        "chunks": [text],
        "document": document,
        "project": project,
    }


class AnswerConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.service = AnswerConfidenceService()

    def test_retrieval_relevance_uses_distance(self):
        high = self.service.calculate_retrieval_relevance([{"distance": 0.05}])
        low = self.service.calculate_retrieval_relevance([{"distance": 0.64}])
        self.assertGreater(high, low)
        self.assertGreater(high, 0.75)

    def test_strong_single_source_answer_scores_high(self):
        citations = [source(1, "The application audits sales order history and generates downloadable reports.")]
        result = self.service.calculate_confidence(
            "The application audits sales order history. [1]\nIt generates downloadable reports. [1]",
            [{"distance": 0.05, "source_id": 1}],
            citations,
            "WAM",
            "WAM.pdf",
        )
        self.assertGreaterEqual(result.score, 85)
        self.assertEqual(result.evidence_consistency, 1.0)
        self.assertEqual(result.citation_coverage, 1.0)

    def test_weak_retrieval_is_not_mistaken_for_strong_support(self):
        relevance = self.service.calculate_retrieval_relevance([{"distance": 0.64}])
        self.assertLess(relevance, 0.5)

    def test_groundedness_and_unsupported_content(self):
        evidence = [source(1, "The application is enabled for administrators.")]
        grounded = self.service.calculate_groundedness("The application is enabled [1].", evidence)
        unsupported = self.service.calculate_groundedness("The application is enabled for customers and exports invoices [1].", evidence)
        self.assertGreater(grounded, unsupported)

    def test_citation_coverage(self):
        citations = [source(1, "The application is enabled.")]
        self.assertEqual(self.service.calculate_citation_coverage("It is enabled [1]. It supports billing.", citations), 0.5)
        self.assertEqual(self.service.calculate_citation_coverage("It is enabled [1].", citations), 1.0)

    def test_partial_and_missing_citations(self):
        citations = [source(1, "The application is enabled and supports billing.")]
        self.assertEqual(self.service.calculate_citation_coverage("It is enabled [1]. It supports billing.", citations), 0.5)
        self.assertEqual(self.service.calculate_citation_coverage("It is enabled.", citations), 0.0)

    def test_consistency_detects_conflict(self):
        consistent = [source(1, "The application is enabled."), source(2, "The application is enabled.", "Other.pdf")]
        conflicting = [source(1, "The application is enabled."), source(2, "The application is disabled.", "Other.pdf")]
        self.assertGreater(self.service.calculate_evidence_consistency(consistent), self.service.calculate_evidence_consistency(conflicting))
        self.assertEqual(self.service.calculate_evidence_consistency([consistent[0]]), 1.0)
        self.assertLess(self.service.calculate_evidence_consistency(conflicting), 1.0)

    def test_unknown_factor_is_renormalized(self):
        class UnknownScopeService(AnswerConfidenceService):
            def calculate_scope_match(self, sources, project=None, document=None):
                return None

        citations = [source(1, "The application is enabled.")]
        result = UnknownScopeService().calculate_confidence(
            "The application is enabled [1].",
            [{"distance": 0.05, "source_id": 1}],
            citations,
        )
        self.assertIsNotNone(result.score)
        self.assertIsNone(result.scope_match)

    def test_scope_and_weights(self):
        citations = [source(1, "The application is enabled.")]
        self.assertEqual(self.service.calculate_scope_match(citations, "WAM", "WAM.pdf"), 1.0)
        with self.assertRaises(ValueError):
            ConfidenceWeights(0.5, 0.3, 0.2, 0.1, 0.1)

    def test_unavailable_without_cited_evidence(self):
        result = self.service.calculate_confidence("An answer.", [], [])
        self.assertIsNone(result.score)
        self.assertEqual(result.level, "Unavailable")

    def test_final_score_and_level(self):
        citations = [source(1, "The application is enabled.")]
        result = self.service.calculate_confidence(
            "The application is enabled [1].",
            [{"distance": 0.05, "source_id": 1, "text": citations[0]["text"]}],
            citations,
            "WAM",
            "WAM.pdf",
        )
        self.assertIsNotNone(result.score)
        self.assertIn(result.level, {"High", "Medium", "Low", "Very Low"})


if __name__ == "__main__":
    unittest.main()
