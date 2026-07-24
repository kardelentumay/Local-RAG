from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
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

    def ingest(
        self,
        path: Path,
        chunk_size: int = 700,
        overlap: int = 100,
        batch_size: int = 16,
        source_root: Path | None = None,
        category: str = "RAG Research",
    ) -> IngestReport:
        if batch_size < 1:
            raise ValueError("batch_size en az 1 olmalıdır")
        processed = skipped = total_chunks = 0
        base = source_root or (path if path.is_dir() else path.parent)
        base = base.resolve()
        for document in discover_documents(path):
            source = document.resolve().relative_to(base).as_posix()
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
            self.store.replace_document(source, digest, chunks, vectors, category)
            processed += 1
            total_chunks += len(chunks)
        return IngestReport(processed, skipped, total_chunks)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        clean = question.strip()
        if not clean:
            raise ValueError("Soru boş olamaz")
        retrieval_query = _normalize_retrieval_query(clean)
        candidate_count = max(top_k * 10, 20)
        results = self._retrieve(retrieval_query, candidate_count, top_k)
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
        scope_term, scoped_sources = _rarest_term_scope(
            self.store, query, candidate_count
        )
        content_query = (
            _without_term(query, scope_term) if scoped_sources else query
        )
        instructed_query = f"{RETRIEVAL_INSTRUCTION}{content_query}"
        query_embedding = self.embeddings.embed([instructed_query])[0]
        vector_results = self.store.search(
            query_embedding, candidate_count, scoped_sources or None
        )
        keyword_results = self.store.keyword_search(
            content_query, candidate_count, scoped_sources or None
        )
        fused = _hybrid_fuse(
            vector_results,
            keyword_results,
            candidate_count if scoped_sources else top_k,
            per_source_limit=candidate_count if scoped_sources else 2,
        )
        if scoped_sources:
            return _rerank_section_matches(content_query, fused, top_k)
        return fused[:top_k]

    def answer(self, question: str, top_k: int = 3) -> Answer:
        if self.chat is None:
            raise RuntimeError("Cevap üretmek için chat sağlayıcısı gerekli")
        retrieval_query = _normalize_retrieval_query(question)
        scope_term, _ = _rarest_term_scope(
            self.store, retrieval_query, max(top_k * 10, 20)
        )
        expanded_top_k = max(top_k, 6) if scope_term and top_k > 1 else top_k
        results = self.search(question, expanded_top_k)
        validation_query = _without_term(retrieval_query, scope_term)
        relevance_results = [
            SearchResult(
                item.source,
                item.position,
                self.store.get_window(item.source, item.position, 1, 2),
                item.score,
            )
            for item in results
        ]
        if not _has_relevant_evidence(
            validation_query,
            relevance_results,
            allow_single_term=bool(scope_term),
        ):
            return Answer(_fallback_for(question), [])
        context = "\n\n".join(
            f"[{index}]\n{_context_excerpt(retrieval_query, self.store.get_window(item.source, item.position, 1, 2), 1500 if scope_term else 1100)}"
            for index, item in enumerate(results, start=1)
        )
        prompt = _build_answer_prompt(question, context)
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
        elif not _is_turkish(question) and _looks_turkish(text):
            text = self.chat.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a translation editor. Translate the supplied answer completely "
                            "into English. Do not add or remove information, and preserve the source "
                            "numbers [1], [2], and [3] exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Write the following text entirely in English:\n\n{text.strip()}",
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


def _looks_turkish(text: str) -> bool:
    lowered = text.lower()
    turkish_characters = sum(lowered.count(character) for character in "çğıöşü")
    words = re.findall(r"[^\W_]+", lowered, flags=re.UNICODE)
    turkish_words = sum(
        word in {"ve", "bir", "bu", "için", "ile", "olarak", "eğitimi", "gibi"}
        for word in words
    )
    return turkish_characters >= 2 or turkish_words >= 3


def _build_answer_prompt(question: str, context: str) -> str:
    if _is_turkish(question):
        return (
            f"BAĞLAM:\n{context}\n\nSORU:\n{question.strip()}\n\n"
            "ÖNEMLİ: Yanıtı yalnızca Türkçe yaz ve bağlamda açıkça desteklenmeyen "
            "hiçbir ayrıntı ekleme."
        )
    return (
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question.strip()}\n\n"
        "IMPORTANT: Answer only in English. Use only facts explicitly supported by "
        "the context. For list questions, extract every item under the relevant section "
        "and return only short bullets; never stop after the first item. Do not expand "
        "abbreviations or invent explanations."
    )


def _normalize_retrieval_query(question: str) -> str:
    normalized = question.strip()
    typo_corrections = {
        r"\bsertificates?\b": "certificates",
        r"\bcertificats?\b": "certificates",
    }
    for pattern, replacement in typo_corrections.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    filler_patterns = (
        r"^\s*(?:please\s+)?(?:give|show|tell)\s+me\s+(?:the\s+)?",
        r"\b(?:the\s+)?information\s+about\b",
        r"\b(?:the\s+)?details\s+about\b",
    )
    for pattern in filler_patterns:
        normalized = re.sub(pattern, " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or question.strip()


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
    per_source_limit: int = 2,
) -> list[SearchResult]:
    scores: dict[tuple[str, int], float] = {}
    items: dict[tuple[str, int], SearchResult] = {}
    semantic_scores = {
        (item.source, item.position): item.score for item in vector_results
    }
    for result_list, weight in ((vector_results, 1.0), (keyword_results, 1.2)):
        for rank, item in enumerate(result_list, start=1):
            key = (item.source, item.position)
            items[key] = item
            scores[key] = scores.get(key, 0.0) + weight / (rank_constant + rank)

    ranked = sorted(scores, key=scores.get, reverse=True)
    selected: list[SearchResult] = []
    source_counts: dict[str, int] = {}
    for key in ranked:
        item = items[key]
        if source_counts.get(item.source, 0) >= per_source_limit:
            continue
        selected.append(
            SearchResult(
                item.source,
                item.position,
                item.content,
                semantic_scores.get(key, 0.0),
            )
        )
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def _rerank_section_matches(
    query: str,
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    query_terms = _meaningful_terms(query)
    if not query_terms:
        return results[:top_k]

    def section_score(item: SearchResult) -> float:
        best = 0.0
        for line in item.content.splitlines():
            clean = line.strip(" \t•-*:#")
            if (
                not clean
                or len(clean) > 80
                or re.search(r"[.!?]$", clean)
                or not clean[0].isupper()
                or "," in clean
            ):
                continue
            line_terms = _meaningful_terms(clean)
            if not line_terms:
                continue
            matched = sum(
                1
                for query_term in query_terms
                if any(
                    _terms_are_close(query_term, line_term)
                    for line_term in line_terms
                )
            )
            coverage = matched / len(query_terms)
            if matched:
                best = max(best, 1.0 + coverage)
        return best

    ranked = sorted(
        enumerate(results),
        key=lambda row: (-section_score(row[1]), row[0]),
    )
    return [item for _, item in ranked[:top_k]]


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
            if any(_terms_are_close(query_term, term) for term in sentence_terms)
        )
        heading_bonus = (
            3
            if len(sentence) <= 80
            and not re.search(r"[.!?]$", sentence)
            and sentence[0].isupper()
            and "," not in sentence
            and (exact_matches or prefix_matches)
            else 0
        )
        ranked.append(
            (exact_matches * 2 + prefix_matches + heading_bonus, index, sentence)
        )
    relevant = [item for item in sorted(ranked, key=lambda row: (-row[0], row[1])) if item[0] > 0]
    if not relevant:
        return " ".join(sentences)[:max_chars]

    _, best_index, best_sentence = relevant[0]
    is_heading = len(best_sentence) <= 80 and not re.search(
        r"[.!?]$", best_sentence
    ) and best_sentence[0].isupper() and "," not in best_sentence
    if is_heading:
        chosen_indices = set(
            range(best_index, min(len(sentences), best_index + 30))
        )
    else:
        chosen_indices = {index for _, index, _ in relevant[:3]}
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


def _has_relevant_evidence(
    question: str,
    results: list[SearchResult],
    allow_single_term: bool = False,
) -> bool:
    if not results:
        return False
    if max(item.score for item in results) >= MIN_ANSWER_RELEVANCE:
        return True

    query_terms = _meaningful_terms(question)
    if not query_terms or (len(query_terms) < 2 and not allow_single_term):
        return False
    evidence_terms = _meaningful_terms(
        " ".join(item.content for item in results)
    )
    matched_terms = {
        query_term
        for query_term in query_terms
        if any(_terms_are_close(query_term, evidence_term) for evidence_term in evidence_terms)
    }
    distinctive_matches = sum(len(term) >= 6 for term in matched_terms)
    if allow_single_term:
        return (
            distinctive_matches >= 1
            and len(matched_terms) / len(query_terms) >= 0.5
        )
    return (
        len(matched_terms) >= 2
        and distinctive_matches >= 2
        and len(matched_terms) / len(query_terms) >= 0.6
    )


def _rarest_term_sources(
    store: SQLiteStore,
    query: str,
    candidate_count: int,
    max_sources: int = 3,
) -> list[str]:
    return _rarest_term_scope(
        store, query, candidate_count, max_sources
    )[1]


def _rarest_term_scope(
    store: SQLiteStore,
    query: str,
    candidate_count: int,
    max_sources: int = 3,
) -> tuple[str, list[str]]:
    candidates: list[tuple[int, int, int, str, list[str]]] = []
    matched_term_count = 0
    for term in _meaningful_terms(query):
        if len(term) < 4:
            continue
        matches = store.keyword_search(term, candidate_count)
        sources = list(dict.fromkeys(item.source for item in matches))
        if sources:
            matched_term_count += 1
        if 0 < len(sources) <= max_sources:
            candidates.append(
                (len(sources), -len(matches), -len(term), term, sources)
            )
    if not candidates or matched_term_count < 2:
        return "", []
    selected = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    return selected[3], selected[4]


def _without_term(query: str, excluded_term: str) -> str:
    if not excluded_term:
        return query
    return re.sub(
        rf"\b{re.escape(excluded_term)}(?:'s)?\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )


def _terms_are_close(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) < 6:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.86


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
