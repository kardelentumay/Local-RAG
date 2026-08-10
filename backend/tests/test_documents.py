import tempfile
import unittest
from pathlib import Path

from local_rag.documents import chunk_text, discover_documents, read_document


class ChunkTextTests(unittest.TestCase):
    def test_short_text_produces_one_chunk(self):
        chunks = chunk_text("Birinci paragraf.\n\nİkinci paragraf.", "note.md", 100, 10)
        self.assertEqual(1, len(chunks))
        self.assertEqual("note.md", chunks[0].source)

    def test_long_text_has_overlap_and_positions(self):
        chunks = chunk_text(" ".join([f"kelime{i}." for i in range(100)]), "note.txt", 120, 20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(list(range(len(chunks))), [item.position for item in chunks])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("metin", "x.txt", 100, 100)

    def test_excel_workbook_is_discovered_and_read_with_coordinates(self):
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sales.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Sales Data"
            sheet.append(["Product", "Units", "Revenue"])
            sheet.append(["Laptop", 8, 256000])
            workbook.save(path)

            self.assertEqual([path], discover_documents(path))
            text = read_document(path)

        self.assertIn("[Çalışma Sayfası: Sales Data]", text)
        self.assertIn("A1=Product", text)
        self.assertIn("B2=8", text)
        self.assertIn("C2=256000", text)


if __name__ == "__main__":
    unittest.main()
