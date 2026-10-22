import unittest

from langchain_core.documents import Document

from src.document_cleaner import clean_document, clean_documents


class DocumentCleanerTests(unittest.TestCase):
    def test_toc_dot_leaders_and_page_numbers_are_removed(self):
        document = Document(
            page_content="""..................................................... 2
2.5. Optimized Supply Chain .................................. 3
2.6. Internal Automation Benefit ............................. 3""",
        )
        cleaned = clean_document(document).page_content
        self.assertEqual(
            cleaned,
            "2.5. Optimized Supply Chain\n2.6. Internal Automation Benefit",
        )

    def test_repeated_headers_and_footers_are_removed(self):
        documents = [
            Document(page_content="MSI SURFACES\nIntroduction\nUseful text\nPage footer"),
            Document(page_content="MSI SURFACES\nRequirements\nMore text\nPage footer"),
        ]
        cleaned = clean_documents(documents)
        self.assertNotIn("MSI SURFACES", cleaned[0].page_content)
        self.assertNotIn("Page footer", cleaned[1].page_content)
        self.assertIn("Useful text", cleaned[0].page_content)

    def test_excessive_whitespace_is_compacted(self):
        document = Document(page_content="  A   useful   sentence.\n\n\n\nB\tvalue  ")
        self.assertEqual(clean_document(document).page_content, "A useful sentence.\n\nB value")

    def test_meaningful_numbers_and_identifiers_are_preserved(self):
        text = "uspWhseCalendar_InsertCapacity\nWAM\nNSWeb\nNS_APP\nVersion 2.5"
        self.assertEqual(clean_document(Document(page_content=text)).page_content, text)

    def test_section_heading_metadata_is_preserved_or_detected(self):
        cleaned = clean_document(Document(page_content="Introduction:\nPurpose and scope"))
        self.assertEqual(cleaned.metadata["section"], "Introduction")
        self.assertEqual(cleaned.metadata["heading"], "Introduction")
        self.assertIn("Purpose and scope", cleaned.page_content)

    def test_valid_table_is_preserved(self):
        text = "| Code | Meaning |\n| WAM | Warehouse Management |"
        self.assertEqual(clean_document(Document(page_content=text)).page_content, text)

    def test_page_number_is_only_removed_when_page_metadata_supports_it(self):
        self.assertEqual(clean_document(Document(page_content="2025\nRevenue" )).page_content, "2025\nRevenue")
        document = Document(page_content="2\nRevenue", metadata={"page": 1})
        self.assertEqual(clean_document(document).page_content, "Revenue")


if __name__ == "__main__":
    unittest.main()

