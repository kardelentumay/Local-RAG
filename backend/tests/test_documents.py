import unittest

from local_rag.documents import chunk_text


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


if __name__ == "__main__":
    unittest.main()
