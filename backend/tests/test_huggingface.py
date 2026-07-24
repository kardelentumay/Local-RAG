import json
import tempfile
import unittest
from pathlib import Path

from local_rag.huggingface import download_articles


class HuggingFaceDownloadTests(unittest.TestCase):
    def test_download_writes_markdown_and_manifest(self):
        payload = {
            "rows": [
                {
                    "row": {
                        "title": "Retrieval-Augmented Generation: A Test",
                        "year": 2024,
                        "paper_id": "RAG_001",
                        "link": "https://example.test/paper",
                        "content": "This paper explains retrieval and generation.",
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rag"
            report = download_articles("rag", 1, output, fetch_json=lambda _: payload)
            articles = list(output.glob("*.md"))
            self.assertEqual(1, report.downloaded)
            self.assertEqual(1, len(articles))
            text = articles[0].read_text(encoding="utf-8")
            self.assertIn("Retrieval-Augmented Generation", text)
            self.assertIn("https://example.test/paper", text)
            manifest = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("rag", manifest["topic"])

    def test_invalid_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                download_articles("rag", 101, Path(directory), fetch_json=lambda _: {})


if __name__ == "__main__":
    unittest.main()
