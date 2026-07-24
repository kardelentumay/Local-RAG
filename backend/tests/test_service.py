import tempfile
import unittest
from pathlib import Path

from local_rag.models import SearchResult
from local_rag.service import (
    FALLBACK_ANSWER_EN,
    FALLBACK_ANSWER_TR,
    RAGService,
    _context_excerpt,
    _build_answer_prompt,
    _has_relevant_evidence,
    _hybrid_fuse,
    _validate_citations,
)
from local_rag.store import SQLiteStore


class KeywordEmbeddings:
    words = ("saat", "bilgisayar", "rag", "kaynak")

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [float(lowered.count(word)) for word in self.words]
            if not any(vector):
                vector.append(1.0)
            else:
                vector.append(0.0)
            vectors.append(vector)
        return vectors


class RecordingChat:
    def __init__(self):
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return "Dersler 09.00'da başlar [1]."


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.docs = root / "docs"
        self.docs.mkdir()
        (self.docs / "program.md").write_text(
            "Dersler hafta içi saat 09.00'da başlar.\n\nKatılımcılar bilgisayar getirmelidir.\n\nRAG cevapları kaynaklara dayanır.",
            encoding="utf-8",
        )
        self.store = SQLiteStore(root / "knowledge.db")
        self.chat = RecordingChat()
        self.service = RAGService(self.store, KeywordEmbeddings(), self.chat)

    def tearDown(self):
        self.temp.cleanup()

    def test_ingest_is_idempotent_and_searches(self):
        first = self.service.ingest(self.docs, chunk_size=100, overlap=10)
        second = self.service.ingest(self.docs, chunk_size=100, overlap=10)
        self.assertEqual(1, first.processed)
        self.assertGreater(first.chunks, 0)
        self.assertEqual(1, second.skipped)
        self.assertEqual("program.md", self.service.search("saat kaçta?", 1)[0].source)

    def test_changed_chunk_settings_force_reindex(self):
        first = self.service.ingest(self.docs, chunk_size=100, overlap=10)
        changed = self.service.ingest(self.docs, chunk_size=120, overlap=20)
        self.assertEqual(1, first.processed)
        self.assertEqual(1, changed.processed)

    def test_ingest_can_keep_source_relative_to_document_root(self):
        uploads = self.docs / "uploads"
        uploads.mkdir()
        uploaded = uploads / "notes.md"
        uploaded.write_text("RAG kaynaklarla yanıt üretir.", encoding="utf-8")
        report = self.service.ingest(
            uploaded,
            source_root=self.docs,
            category="Course Notes",
        )
        self.assertEqual(1, report.processed)
        sources = {item["source"] for item in self.store.list_documents()}
        self.assertIn("uploads/notes.md", sources)
        uploaded_item = next(
            item for item in self.store.list_documents()
            if item["source"] == "uploads/notes.md"
        )
        self.assertEqual("Course Notes", uploaded_item["category"])

    def test_store_lists_and_deletes_document_with_chunks(self):
        self.service.ingest(self.docs, chunk_size=100, overlap=10)
        listed = self.store.list_documents()
        self.assertEqual("program.md", listed[0]["source"])
        self.assertGreater(listed[0]["chunks"], 0)
        self.assertTrue(self.store.delete_document("program.md"))
        self.assertFalse(self.store.delete_document("program.md"))
        self.assertEqual((0, 0), self.store.stats())
        self.assertEqual([], self.store.keyword_search("bilgisayar"))

    def test_fts_keyword_search_finds_ingested_content(self):
        self.service.ingest(self.docs, chunk_size=100, overlap=10)
        results = self.store.keyword_search("bilgisayar", 5)
        self.assertTrue(results)
        self.assertEqual("program.md", results[0].source)

    def test_store_returns_previous_chunk_window(self):
        self.service.ingest(self.docs, chunk_size=100, overlap=10)
        window = self.store.get_window("program.md", 1, before=1)
        self.assertIn("Dersler", window)
        self.assertIn("bilgisayar", window)

    def test_answer_includes_retrieved_context(self):
        self.service.ingest(self.docs, chunk_size=100, overlap=10)
        answer = self.service.answer("Dersler saat kaçta?", 1)
        self.assertIn("[1]", answer.text)
        self.assertIn("BAĞLAM", self.chat.messages[1]["content"])
        self.assertNotIn("Kaynak:", self.chat.messages[1]["content"])
        self.assertEqual(1, len(answer.sources))

    def test_search_adds_qwen_retrieval_instruction(self):
        class RecordingEmbeddings(KeywordEmbeddings):
            def __init__(self):
                self.last_text = ""

            def embed(self, texts):
                self.last_text = texts[0]
                return super().embed(texts)

        embeddings = RecordingEmbeddings()
        service = RAGService(self.store, embeddings, self.chat)
        service.ingest(self.docs, chunk_size=100, overlap=10)
        service.search("RAG nedir?", 1)
        self.assertIn("Instruct:", embeddings.last_text)
        self.assertIn("Query: RAG nedir?", embeddings.last_text)

    def test_english_answer_to_turkish_question_is_translated(self):
        class TranslatingChat:
            def __init__(self):
                self.calls = 0

            def complete(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return "The retrieval module is connected to the generation module with context."
                return "Retrieval modülü, bağlam aracılığıyla üretim modülüne bağlanır."

        chat = TranslatingChat()
        service = RAGService(self.store, KeywordEmbeddings(), chat)
        service.ingest(self.docs, chunk_size=100, overlap=10)
        answer = service.answer("RAG nedir ve nasıl çalışır?", 1)
        self.assertEqual(2, chat.calls)
        self.assertIn("üretim", answer.text)

    def test_apology_refusal_is_normalized_and_sources_hidden(self):
        class RefusingChat:
            def complete(self, messages):
                return "Bağlamın desteklemediği için özür dilerim. Başka bir konuda yardımcı olabilirim."

        service = RAGService(self.store, KeywordEmbeddings(), RefusingChat())
        service.ingest(self.docs, chunk_size=100, overlap=10)
        answer = service.answer("RAG nedir?", 1)
        self.assertEqual(FALLBACK_ANSWER_TR, answer.text)
        self.assertEqual([], answer.sources)

    def test_english_fallback_does_not_receive_citation(self):
        class EnglishRefusingChat:
            def complete(self, messages):
                return "No relevant information was found in the loaded documents."

        service = RAGService(self.store, KeywordEmbeddings(), EnglishRefusingChat())
        service.ingest(self.docs, chunk_size=100, overlap=10)
        answer = service.answer("What is quantum computing?", 1)
        self.assertEqual(FALLBACK_ANSWER_EN, answer.text)
        self.assertEqual([], answer.sources)

    def test_context_excerpt_selects_query_relevant_sentences(self):
        content = (
            "RAG is used in many applications. "
            "Standard RAG has limited contextual awareness and can produce fragmented outputs. "
            "Static workflows struggle with multi-step reasoning, scalability, and latency. "
            "Future work includes virtual reality."
        )
        excerpt = _context_excerpt("What are the limitations of standard RAG systems?", content)
        self.assertIn("fragmented outputs", excerpt)
        self.assertNotIn("virtual reality", excerpt)

    def test_repetitive_answer_is_rejected(self):
        class RepeatingChat:
            def complete(self, messages):
                return "Açıklama " + "genellikle " * 30

        service = RAGService(self.store, KeywordEmbeddings(), RepeatingChat())
        service.ingest(self.docs, chunk_size=100, overlap=10)
        answer = service.answer("RAG nedir?", 1)
        self.assertEqual(FALLBACK_ANSWER_TR, answer.text)
        self.assertEqual([], answer.sources)

    def test_hybrid_fusion_rewards_results_found_by_both_retrievers(self):
        vector = [
            SearchResult("vector-only.md", 1, "Vector result", 0.9),
            SearchResult("shared.md", 2, "Shared result", 0.8),
        ]
        keyword = [
            SearchResult("shared.md", 2, "Shared result", 2.0),
            SearchResult("keyword-only.md", 3, "Keyword result", 1.0),
        ]
        results = _hybrid_fuse(vector, keyword, 2)
        self.assertEqual("shared.md", results[0].source)
        self.assertEqual(0.8, results[0].score)

    def test_hybrid_fusion_keeps_semantic_confidence_for_keyword_match(self):
        vector = [
            SearchResult("paper.md", 1, "Generic model value table.", 0.21),
        ]
        keyword = [
            SearchResult("paper.md", 1, "Generic model value table.", 9.0),
        ]
        results = _hybrid_fuse(vector, keyword, 1)
        self.assertEqual(0.21, results[0].score)

    def test_keyword_only_result_beats_equally_ranked_vector_only_result(self):
        vector = [
            SearchResult("generic.md", 1, "Generic vector result", 0.50),
        ]
        keyword = [
            SearchResult("cv.pdf", 9, "Kardelen certificate list", 8.0),
        ]
        results = _hybrid_fuse(vector, keyword, 2)
        self.assertEqual("cv.pdf", results[0].source)

    def test_off_topic_question_is_rejected_before_chat_generation(self):
        class UnexpectedChat:
            def complete(self, messages):
                raise AssertionError("Chat generation must not run for an unrelated question.")

        service = RAGService(self.store, KeywordEmbeddings(), UnexpectedChat())
        service.ingest(self.docs, chunk_size=100, overlap=10)
        answer = service.answer("What is the Bitcoin value today?", 2)
        self.assertEqual(FALLBACK_ANSWER_EN, answer.text)
        self.assertEqual([], answer.sources)

    def test_low_semantic_score_is_accepted_with_strong_fuzzy_text_evidence(self):
        results = [
            SearchResult(
                "cv.pdf",
                8,
                "Kardelen Tumay\nSertificates\nArtificial Intelligence Training",
                0.42,
            )
        ]
        self.assertTrue(
            _has_relevant_evidence("What are Kardelen's certificates?", results)
        )

    def test_generic_word_overlap_does_not_admit_off_topic_question(self):
        results = [
            SearchResult(
                "evaluation.md",
                3,
                "The best value in each model category is shown in bold.",
                0.50,
            )
        ]
        self.assertFalse(
            _has_relevant_evidence("What is the Bitcoin value today?", results)
        )

    def test_context_excerpt_removes_unrelated_sentences(self):
        content = (
            "Brain-computer interfaces enable immersive applications. "
            "Incorrect retrieved knowledge can cause hallucinations in RAG systems. "
            "Corrective retrieval improves grounding and factuality. "
            "Virtual reality is another future research direction."
        )
        excerpt = _context_excerpt("How can hallucinations be reduced?", content)
        self.assertIn("cause hallucinations", excerpt)
        self.assertNotIn("Brain-computer", excerpt)

    def test_context_excerpt_selects_agentic_comparison_sentences(self):
        content = (
            "RAG is used in many industries. "
            "Traditional RAG systems use static workflows and have limited adaptability. "
            "Unlike traditional RAG, Agentic RAG uses autonomous agents, adaptive retrieval, and iterative refinement. "
            "Healthcare is one possible application."
        )
        excerpt = _context_excerpt("How does agentic RAG differ from traditional RAG?", content)
        self.assertIn("static workflows", excerpt)
        self.assertIn("autonomous agents", excerpt)
        self.assertNotIn("Healthcare", excerpt)

    def test_context_excerpt_recovers_minor_heading_misspelling(self):
        content = (
            "Languages: Turkish and English. "
            "Sertificates: Artificial Intelligence Training and RPA Training. "
            "Kardelen Tumay - CV."
        )
        excerpt = _context_excerpt("What are Kardelen's certificates?", content)
        self.assertIn("Artificial Intelligence Training", excerpt)

    def test_english_question_uses_fully_english_answer_prompt(self):
        prompt = _build_answer_prompt(
            "What are Kardelen's certificates?",
            "Sertificates: Artificial Intelligence Training.",
        )
        self.assertIn("CONTEXT:", prompt)
        self.assertIn("QUESTION:", prompt)
        self.assertIn("Answer only in English", prompt)
        self.assertNotIn("BAĞLAM", prompt)

    def test_citation_validator_removes_out_of_range_numbers(self):
        answer = _validate_citations("First claim [1]. Invalid claim [3]. Second [2].", 2)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)
        self.assertNotIn("[3]", answer)

    def test_citation_validator_adds_first_source_when_missing(self):
        answer = _validate_citations("A grounded answer.", 2)
        self.assertEqual("A grounded answer. [1]", answer)

    def test_citation_validator_removes_all_citations_without_sources(self):
        answer = _validate_citations("Unsupported [1] statement.", 0)
        self.assertEqual("Unsupported statement.", answer)

    def test_citation_validator_collapses_repeated_citations(self):
        answer = _validate_citations("A claim [1] [1] [1] [1] [", 2)
        self.assertEqual("A claim [1]", answer)

    def test_citation_validator_removes_standalone_leading_citation(self):
        answer = _validate_citations("[1]\nA grounded comparison. [1]", 1)
        self.assertEqual("A grounded comparison. [1]", answer)

    def test_citation_validator_removes_standalone_citation_inside_answer(self):
        answer = _validate_citations("First paragraph.\n\n[2]\nSecond paragraph. [1] [2]", 2)
        self.assertEqual("First paragraph.\nSecond paragraph. [1] [2]", answer)


if __name__ == "__main__":
    unittest.main()
