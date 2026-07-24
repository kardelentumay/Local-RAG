import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_rag import api
from local_rag.models import Answer, SearchResult


class FakeRuntime:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.documents_path = db_path.parent / "documents"
        self.models_loaded = False

    def stats(self):
        return 20, 1692

    def ask(self, question: str, top_k: int):
        self.models_loaded = True
        return Answer(
            text=f"Grounded answer for {question} [1]",
            sources=[
                SearchResult(
                    source="paper.md",
                    position=4,
                    content="A grounded source passage.",
                    score=0.91234,
                )
            ][:top_k],
        )

    def list_documents(self):
        return [
            {
                "source": "uploads/paper.md",
                "category": "RAG Research",
                "updated_at": "2026-07-24 10:00:00",
                "chunks": 7,
            }
        ]

    def delete_document(self, source: str):
        return source == "uploads/paper.md"


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fake_runtime = FakeRuntime(Path(self.temp.name) / "knowledge.db")
        self.runtime_patch = patch.object(api, "runtime", self.fake_runtime)
        self.runtime_patch.start()

    def tearDown(self):
        self.runtime_patch.stop()
        self.temp.cleanup()

    def test_health_does_not_load_models(self):
        response = asyncio.run(api.health())
        self.assertEqual("ok", response.status)
        self.assertEqual("local", response.mode)
        self.assertFalse(response.models_loaded)

    def test_status_returns_database_counts(self):
        response = asyncio.run(api.status())
        self.assertEqual(20, response.documents)
        self.assertEqual(1692, response.chunks)
        self.assertEqual(str(self.fake_runtime.db_path), response.database)

    def test_ask_returns_numbered_sources(self):
        response = asyncio.run(api.ask(api.AskRequest(question="What is RAG?", top_k=2)))
        self.assertIn("[1]", response.answer)
        self.assertEqual(1, len(response.sources))
        self.assertEqual(1, response.sources[0].number)
        self.assertEqual("paper.md", response.sources[0].document)
        self.assertEqual(5, response.sources[0].chunk)
        self.assertEqual(0.9123, response.sources[0].score)

    def test_documents_returns_indexed_files(self):
        response = asyncio.run(api.documents())
        self.assertEqual(1, len(response))
        self.assertEqual("paper.md", response[0].name)
        self.assertEqual("MD", response[0].kind)
        self.assertEqual("RAG Research", response[0].category)
        self.assertEqual(7, response[0].chunks)

    def test_uploaded_document_hides_storage_identifier(self):
        item = api._document_response(
            {
                "source": "uploads/1234567890abcdef1234567890abcdef-notes.md",
                "category": "Course Notes",
                "updated_at": "2026-07-24 10:00:00",
                "chunks": 1,
            }
        )
        self.assertEqual("notes.md", item.name)
        self.assertEqual("Course Notes", item.category)

    def test_category_name_is_normalized(self):
        self.assertEqual("Course Notes", api._normalize_category("  Course   Notes  "))

    def test_delete_document_returns_not_found_for_unknown_source(self):
        with self.assertRaises(api.HTTPException) as context:
            asyncio.run(api.delete_document("unknown.md"))
        self.assertEqual(404, context.exception.status_code)


if __name__ == "__main__":
    unittest.main()
