import unittest

from src.search import RAGSearch


class AnswerFormattingTests(unittest.TestCase):
    def test_citations_are_sequential_and_deduplicated(self):
        answer, sources, mapping = RAGSearch._normalize_citations(
            "### Objectives\n1. Track inventory [1]\n2. Improve accuracy [3]\n3. Track inventory [1]",
            [
                {"id": 1, "document": "one.pdf"},
                {"id": 2, "document": "unused.pdf"},
                {"id": 3, "document": "three.pdf"},
            ],
        )
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)
        self.assertNotIn("[3]", answer)
        self.assertEqual([source["id"] for source in sources], [1, 2])
        self.assertEqual([source["document"] for source in sources], ["one.pdf", "three.pdf"])
        self.assertEqual(mapping, {1: 1, 3: 2})

    def test_unknown_citations_are_removed(self):
        answer, sources, _ = RAGSearch._normalize_citations(
            "Supported claim [1]. Unsupported claim [9].",
            [{"id": 1, "document": "one.pdf"}],
        )
        self.assertEqual(answer, "Supported claim [1]. Unsupported claim .")
        self.assertEqual(len(sources), 1)


if __name__ == "__main__":
    unittest.main()

