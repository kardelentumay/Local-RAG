from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .documents import chunk_text, discover_documents, read_document
from .foundry import ChatProvider, EmbeddingProvider
from .models import Answer, SearchResult
from .store import SQLiteStore


SYSTEM_PROMPT = """You are a source-grounded assistant for academic papers.
Rules:
- Answer using only statements supported by the CONTEXT. Do not use outside knowledge.
- Do not add unsupported organizations, people, products, dates, or features.
- If the context is insufficient, say exactly: 'No relevant information was found in the loaded documents.'
- Answer in the same language as the user's question.
- End each factual sentence with the supporting source number such as [1] or [2].
- Source numbers must match the numbered passages in the CONTEXT.
- Do not repeat passage labels, filenames, prompt instructions, or comments about your own answer.
- Output only the answer itself, followed by inline source numbers.
- Keep the answer concise, direct, and no longer than four paragraphs.
"""

RETRIEVAL_INSTRUCTION = (
    "Instruct: Given a question in any language, retrieve relevant English passages "
    "from research papers that answer the question\nQuery: "
)

RETRY_SYSTEM_PROMPT = """You are a source-grounded assistant for academic papers.
The retrieval system has already selected relevant passages. Answer from those passages only.
If the question asks for a comparison and the passages describe the subjects separately, compare
their stated roles or properties without inventing facts. If the passages support only part of the
question, answer that supported part and state the limitation briefly instead of refusing entirely.
Answer in the same language as the question, cite claims with [1], [2], or [3], and be concise.
"""

FALLBACK_ANSWER_TR = "Bu bilgi yüklenen belgelerde bulunmuyor."
FALLBACK_ANSWER_EN = "No relevant information was found in the loaded documents."
MIN_ANSWER_RELEVANCE = 0.55


@dataclass(frozen=True)
class IngestReport:
    processed: int
    skipped: int
    chunks: int


class RAGService:
    def __init__(self, store: SQLiteStore, embeddings: EmbeddingProvider, chat: ChatProvider | None = None):
        self.store = store
        self.embeddings = embeddings
        self.chat = chat

    def ingest(self, path: Path, chunk_size: int = 700, overlap: int = 100, batch_size: int = 16) -> IngestReport:
        if batch_size < 1:
            raise ValueError("batch_size en az 1 olmalıdır")
        processed = skipped = total_chunks = 0
        base = path if path.is_dir() else path.parent
        for document in discover_documents(path):
            source = document.relative_to(base).as_posix()
            raw = document.read_bytes()
            index_version = f"hybrid-v1:{chunk_size}:{overlap}".encode("ascii")
            digest = hashlib.sha256(raw + index_version).hexdigest()
            if self.store.has_document(source, digest):
                skipped += 1
                continue
            chunks = chunk_text(read_document(document), source, chunk_size, overlap)
            vectors: list[list[float]] = []
            for start in range(0, len(chunks), batch_size):
                vectors.extend(self.embeddings.embed([item.content for item in chunks[start : start + batch_size]]))
            self.store.replace_document(source, digest, chunks, vectors)
            processed += 1
            total_chunks += len(chunks)
        return IngestReport(processed, skipped, total_chunks)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        clean = question.strip()
        if not clean:
            raise ValueError("Soru boş olamaz")
        candidate_count = max(top_k * 10, 20)
        results = self._retrieve(clean, candidate_count, top_k)
        facets = _comparison_facets(clean)
        if len(facets) != 2 or top_k < 2:
            return results

        balanced: list[SearchResult] = []
        seen: set[tuple[str, int]] = set()
        for facet in facets:
            facet_results = self._retrieve(facet, candidate_count, 1)
            if facet_results:
                item = facet_results[0]
                key = (item.source, item.position)
                if key not in seen:
                    balanced.append(item)
                    seen.add(key)
        for item in results:
            key = (item.source, item.position)
            if key not in seen:
                balanced.append(item)
                seen.add(key)
            if len(balanced) == top_k:
                break
        return balanced[:top_k]

    def _retrieve(self, query: str, candidate_count: int, top_k: int) -> list[SearchResult]:
        instructed_query = f"{RETRIEVAL_INSTRUCTION}{query}"
        vector_results = self.store.search(self.embeddings.embed([instructed_query])[0], candidate_count)
        keyword_results = self.store.keyword_search(query, candidate_count)
        return _hybrid_fuse(vector_results, keyword_results, top_k)

    def answer(self, question: str, top_k: int = 3) -> Answer:
        if self.chat is None:
            raise RuntimeError("Cevap üretmek için chat sağlayıcısı gerekli")
        results = self.search(question, top_k)
        if not results or max(item.score for item in results) < MIN_ANSWER_RELEVANCE:
            return Answer(_fallback_for(question), [])
        context = "\n\n".join(
            f"[{index}]\n{_context_excerpt(question, self.store.get_window(item.source, item.position, 1, 1), 1100)}"
            for index, item in enumerate(results, start=1)
        )
        prompt = (
            f"BAĞLAM:\n{context}\n\nSORU:\n{question.strip()}\n\n"
            "ÖNEMLİ: Yanıtı sorunun dilinde yaz ve bağlamda açıkça desteklenmeyen hiçbir ayrıntı ekleme."
        )
        text = self.chat.complete([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        if _is_refusal(text.strip()):
            text = self.chat.complete(
                [{"role": "system", "content": RETRY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            )
        if _is_turkish(question) and _looks_english(text):
            text = self.chat.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Sen yalnızca çeviri yapan bir editörsün. Verilen yanıtı eksiksiz Türkçeye çevir. "
                            "Yeni bilgi ekleme, bilgiyi çıkarma ve [1], [2], [3] kaynak numaralarını aynen koru."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Aşağıdaki metnin tamamını yalnızca Türkçe yaz:\n\n{text.strip()}",
                    },
                ]
            )
        clean_answer = text.strip()
        if _is_refusal(clean_answer) or _has_degenerate_repetition(clean_answer):
            return Answer(_fallback_for(question), [])
        clean_answer = _validate_citations(clean_answer, len(results))
        return Answer(clean_answer, results)

def _is_turkish(text: str) -> bool:
    lowered = text.lower()
    if any(character in lowered for character in "çğıöşü"):
        return True
    words = set(re.findall(r"[a-zA-Z]+", lowered))
    return bool(words & {"nedir", "nelerdir", "nasıl", "neden", "hangi", "açıkla", "ve", "için"})


def _fallback_for(question: str) -> str:
    return FALLBACK_ANSWER_TR if _is_turkish(question) else FALLBACK_ANSWER_EN


def _looks_english(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    english = sum(word in {"the", "and", "is", "are", "of", "to", "from", "with", "this", "that"} for word in words)
    turkish = sum(word in {"ve", "bir", "bu", "için", "ile", "olarak", "nedir", "olan"} for word in words)
    return english >= 3 and english > turkish * 2


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    signals = (
        "özür dilerim",
        "yüklenen belgelerde bulunmuyor",
        "bağlamda bulunmuyor",
        "bağlamın desteklemediği",
        "bağlamın desteklemeyeceği",
        "başka bir konuda",
        "no relevant information was found",
        "insufficient context",
        "context does not contain",
        "cannot answer from the context",
    )
    return any(signal in lowered for signal in signals)


def _has_degenerate_repetition(text: str) -> bool:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < 20:
        return False
    _, most_common_count = Counter(words).most_common(1)[0]
    if most_common_count / len(words) >= 0.18:
        return True
    four_grams = list(zip(words, words[1:], words[2:], words[3:]))
    return bool(four_grams) and len(set(four_grams)) / len(four_grams) < 0.55


def _hybrid_fuse(
    vector_results: list[SearchResult],
    keyword_results: list[SearchResult],
    top_k: int,
    rank_constant: int = 60,
) -> list[SearchResult]:
    scores: dict[tuple[str, int], float] = {}
    items: dict[tuple[str, int], SearchResult] = {}
    for result_list in (vector_results, keyword_results):
        for rank, item in enumerate(result_list, start=1):
            key = (item.source, item.position)
            items[key] = item
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)

    max_score = 2.0 / (rank_constant + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    selected: list[SearchResult] = []
    source_counts: dict[str, int] = {}
    for key in ranked:
        item = items[key]
        if source_counts.get(item.source, 0) >= 2:
            continue
        selected.append(
            SearchResult(item.source, item.position, item.content, min(scores[key] / max_score, 1.0))
        )
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def _context_excerpt(query: str, content: str, max_chars: int = 700) -> str:
    query_terms = _meaningful_terms(query)
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", content) if item.strip()]
    if not sentences:
        return content[:max_chars]
    if not query_terms:
        return " ".join(sentences)[:max_chars]

    ranked = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _meaningful_terms(sentence)
        exact_matches = len(query_terms & sentence_terms)
        prefix_matches = sum(
            1 for query_term in query_terms
            if any(term.startswith(query_term[:5]) or query_term.startswith(term[:5]) for term in sentence_terms)
        )
        ranked.append((exact_matches * 2 + prefix_matches, index, sentence))
    relevant = [item for item in sorted(ranked, key=lambda row: (-row[0], row[1])) if item[0] > 0]
    if not relevant:
        return " ".join(sentences)[:max_chars]

    relevant_indices = [index for _, index, _ in relevant[:3]]
    chosen_indices = set(range(min(relevant_indices), max(relevant_indices) + 1))
    excerpt = " ".join(sentences[index] for index in sorted(chosen_indices))
    return excerpt[:max_chars]


def _meaningful_terms(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when",
        "where", "which", "who", "why", "with", "system", "systems",
    }
    return {
        token for token in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        if len(token) >= 3 and token not in stopwords
    }


def _comparison_facets(question: str) -> list[str]:
    english = re.search(
        r"\b(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if english:
        return [english.group(1).strip(), english.group(2).strip()]
    turkish = re.search(
        r"(.+?)\s+ile\s+(.+?)\s+arasındaki\s+fark",
        question,
        flags=re.IGNORECASE,
    )
    if turkish:
        return [turkish.group(1).strip(), turkish.group(2).strip()]
    return []


def _validate_citations(text: str, source_count: int) -> str:
    text = re.sub(r"(?m)^\s*\[\d+\]\s*(?:\r?\n|$)", "", text)
    if source_count < 1:
        return re.sub(r"\s*\[\d+\]", "", text).strip()

    valid_numbers: list[int] = []

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= source_count:
            valid_numbers.append(number)
            return f"[{number}]"
        return ""

    cleaned = re.sub(r"\[(\d+)\]", replace, text)
    cleaned = re.sub(r"\[(\d+)\](?:\s*\[\1\])+", lambda match: f"[{match.group(1)}]", cleaned)
    cleaned = re.sub(r"\[\s*$", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned).strip()
    if not valid_numbers:
        cleaned = f"{cleaned} [1]".strip()
    return cleaned
