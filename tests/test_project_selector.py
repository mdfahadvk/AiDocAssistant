import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from src.project_config import DEFAULT_PROJECT, assign_document_to_project, project_for_document
from src.search import RAGSearch
from src.vectorstore import ChromaVectorStore


class FakeCollection:
    def __init__(self, metadatas):
        self.metadatas = metadatas
        self.where_calls = []

    def count(self):
        return len(self.metadatas)

    def get(self, include=None):
        return {"metadatas": self.metadatas}

    def query(self, **kwargs):
        self.where_calls.append(kwargs.get("where"))
        return {"documents": [["WAM evidence"]], "metadatas": [[{"project": "WAM", "source_label": "WAM.pdf"}]], "distances": [[0.1]]}


class FakeEmbedding:
    def __getitem__(self, index):
        return self

    def tolist(self):
        return [0.0]


class ProjectSelectorTests(unittest.TestCase):
    def test_mapping_resolves_document_and_defaults_unmapped(self):
        mapping = {"WAM": ["WAM_UserGuide.pdf"]}
        self.assertEqual(project_for_document("WAM_UserGuide.pdf", mapping), "WAM")
        self.assertEqual(project_for_document("Architecture.md", mapping), DEFAULT_PROJECT)

    def test_upload_assignment_moves_filename_between_projects(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"PROJECT_MAPPING_FILE": f"{directory}/mapping.json"}
        ):
            with open(f"{directory}/mapping.json", "w", encoding="utf-8") as handle:
                json.dump({"Old": ["guide.pdf"]}, handle)
            mapping = assign_document_to_project("guide.pdf", "WAM")
            self.assertEqual(mapping, {"WAM": ["guide.pdf"]})

    def test_index_summary_groups_projects_and_legacy_rows(self):
        store = object.__new__(ChromaVectorStore)
        store.collection = FakeCollection([
            {"project": "WAM", "source_label": "WAM.pdf"},
            {"source_label": "legacy.pdf"},
        ])
        self.assertEqual(store.project_summary(), {"General / Unknown": ["legacy.pdf"], "WAM": ["WAM.pdf"]})

    def test_filtered_search_passes_project_and_document_to_chroma(self):
        collection = FakeCollection([{"project": "WAM", "source_label": "WAM.pdf"}])
        store = object.__new__(ChromaVectorStore)
        store.collection = collection
        store._model = SimpleNamespace(encode=lambda *args, **kwargs: FakeEmbedding())
        rag = object.__new__(RAGSearch)
        rag.vectorstore = store
        rag.llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="Answer [1]"))
        with patch.dict("os.environ", {"RETRIEVAL_MAX_DISTANCE": "0.65"}):
            result = rag.search_and_summarize("What is WAM?", project="WAM", document="WAM.pdf")
        self.assertEqual(result["sources"][0]["document"], "WAM.pdf")
        self.assertEqual(collection.where_calls[0], {"$and": [{"project": "WAM"}, {"source_label": "WAM.pdf"}]})


if __name__ == "__main__":
    unittest.main()
