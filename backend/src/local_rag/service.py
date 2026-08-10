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
- For comparisons, state only supported contrasts, use each subject's own definition, and do not
  repeat the same distinction in different words.
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

    def search(
        self,
        question: str,
        top_k: int = 3,
        sources: list[str] | None = None,
    ) -> list[SearchResult]:
        clean = question.strip()
        if not clean:
            raise ValueError("Soru boş olamaz")
        retrieval_query = _normalize_retrieval_query(clean)
        candidate_count = max(top_k * 10, 20)
        results = self._retrieve(
            retrieval_query,
            candidate_count,
            top_k,
            use_scope=not _has_named_anchor(clean),
            allowed_sources=sources,
        )
        facets = _comparison_facets(clean)
        if len(facets) != 2 or top_k < 2:
            return results

        balanced: list[SearchResult] = []
        seen: set[tuple[str, int]] = set()
        for facet in facets:
            facet_results = self._retrieve(
                facet, candidate_count, 1, allowed_sources=sources
            )
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

    def _retrieve(
        self,
        query: str,
        candidate_count: int,
        top_k: int,
        use_scope: bool = True,
        allowed_sources: list[str] | None = None,
    ) -> list[SearchResult]:
        if allowed_sources is not None:
            if not allowed_sources:
                return []
            instructed_query = f"{RETRIEVAL_INSTRUCTION}{query}"
            query_embedding = self.embeddings.embed([instructed_query])[0]
            vector_results = self.store.search(
                query_embedding, candidate_count, allowed_sources
            )
            keyword_results = self.store.keyword_search(
                query, candidate_count, allowed_sources
            )
            fused = _hybrid_fuse(
                vector_results,
                keyword_results,
                candidate_count,
                per_source_limit=candidate_count,
            )
            return _rerank_section_matches(query, fused, top_k)

        scope_term, scoped_sources = _rarest_term_scope(
            self.store, query, candidate_count
        ) if use_scope else ("", [])
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
            return _rerank_section_matches(query, fused, top_k)
        return fused[:top_k]

    def answer(
        self,
        question: str,
        top_k: int = 3,
        sources: list[str] | None = None,
    ) -> Answer:
        if self.chat is None:
            raise RuntimeError("Cevap üretmek için chat sağlayıcısı gerekli")
        comparison_facets = _comparison_facets(question)
        if comparison_facets and sources:
            glossary_answer = self._glossary_comparison_answer(
                comparison_facets, sources
            )
            if glossary_answer is not None:
                return glossary_answer
        retrieval_query = _normalize_retrieval_query(question)
        scope_term, _ = _rarest_term_scope(
            self.store, retrieval_query, max(top_k * 10, 20)
        )
        expanded_top_k = (
            max(top_k, 6)
            if top_k > 1
            and not comparison_facets
            and not _has_named_anchor(question)
            and (scope_term or len(_meaningful_terms(retrieval_query)) >= 3)
            else top_k
        )
        if _has_named_anchor(question):
            results = self._retrieve(
                retrieval_query,
                40,
                expanded_top_k,
                use_scope=False,
                allowed_sources=sources,
            )
        else:
            results = self.search(question, expanded_top_k, sources)

        if comparison_facets:
            focused_results = _filter_comparison_results(comparison_facets, results)
            if focused_results:
                results = focused_results[: max(top_k, len(comparison_facets))]

        requested_profile_section = _requested_profile_section(question)
        if requested_profile_section and sources:
            result_sources = {result.source for result in results}
            for source in sources:
                if source in result_sources:
                    continue
                content = self.store.get_document_content(source)
                if content:
                    results.append(SearchResult(source, 0, content[:700], 1.0))

        structured = self._structured_document_answer(question, results)
        if structured is not None:
            return structured

        if _is_project_list_question(question) and scope_term:
            scoped_sources = _rarest_term_sources(
                self.store, retrieval_query, max(top_k * 10, 20)
            )
            if scoped_sources:
                project_section = _extract_section(
                    self.store.get_document_content(scoped_sources[0]),
                    "Projects",
                    ("Experience", "Education", "Skills", "Certificates"),
                )
                if project_section:
                    results = [
                        SearchResult(scoped_sources[0], 0, project_section, 1.0)
                    ]
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
        has_evidence = (
            _has_comparison_evidence(comparison_facets, relevance_results)
            if comparison_facets
            else _has_relevant_evidence(
                validation_query,
                relevance_results,
                allow_single_term=bool(scope_term),
                allow_named_anchor=_has_named_anchor(question),
            )
        )
        if not has_evidence:
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
        if _is_project_list_question(question) and _needs_project_completion(text):
            text = self.chat.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract the complete project list from the context. Return each "
                            "project as one bullet: project name — one short description. "
                            "Use only the context and include every project."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        clean_answer = text.strip()
        if _is_refusal(clean_answer) or _has_degenerate_repetition(clean_answer):
            return Answer(_fallback_for(question), [])
        clean_answer = _validate_citations(clean_answer, len(results))
        return Answer(clean_answer, results)

    def _glossary_comparison_answer(
        self, facets: list[str], sources: list[str]
    ) -> Answer | None:
        definitions: list[tuple[str, str, str]] = []
        for facet in facets:
            match: tuple[str, str] | None = None
            for source in sources:
                definition = _extract_glossary_definition(
                    self.store.get_document_content(source), facet
                )
                if definition:
                    match = (source, definition)
                    break
            if match is None:
                return None
            definitions.append((facet, match[0], match[1]))

        used_sources = list(dict.fromkeys(source for _, source, _ in definitions))
        evidence: list[SearchResult] = []
        for source in used_sources:
            terms = " ".join(
                facet
                for facet, definition_source, _ in definitions
                if definition_source == source
            )
            matches = self.store.keyword_search(terms, 5, [source])
            evidence.append(
                matches[0]
                if matches
                else SearchResult(
                    source,
                    0,
                    self.store.get_document_content(source)[:700],
                    1.0,
                )
            )

        source_numbers = {
            result.source: index for index, result in enumerate(evidence, start=1)
        }
        lines = [
            f"- {facet.title()}: {definition} [{source_numbers[source]}]"
            for facet, source, definition in definitions
        ]
        return Answer("\n".join(lines), evidence)

    def _structured_document_answer(
        self, question: str, results: list[SearchResult]
    ) -> Answer | None:
        if not results:
            return None

        lowered_question = question.lower()
        unique_results = list(
            {
                result.source: result
                for result in results
            }.values()
        )

        for result in unique_results:
            if Path(result.source).suffix.lower() != ".xlsx":
                continue
            analysis_answer = _analyze_spreadsheet(
                question, self.store.get_document_content(result.source)
            )
            if analysis_answer:
                return Answer(f"{analysis_answer} [1]", [result])
            summary_answer = _extract_spreadsheet_summary(
                question, self.store.get_document_content(result.source)
            )
            if summary_answer:
                return Answer(f"{summary_answer} [1]", [result])

        if _is_person_identity_question(question):
            for result in unique_results:
                profile_answer = _person_profile_answer(
                    question, self.store.get_document_content(result.source)
                )
                if profile_answer:
                    return Answer(f"{profile_answer} [1]", [result])

        requested_section = _requested_profile_section(question)
        if requested_section:
            heading, following_headings = requested_section
            for result in unique_results:
                section = _extract_section(
                    self.store.get_document_content(result.source),
                    heading,
                    following_headings,
                )
                formatted = _format_profile_section(heading, section)
                if formatted:
                    return Answer(f"{formatted} [1]", [result])

        if _is_project_list_question(question):
            for result in unique_results:
                section = _extract_section(
                    self.store.get_document_content(result.source),
                    "Projects",
                    ("Experience", "Education", "Skills", "Certificates", "Sertificates"),
                )
                projects = _extract_projects(section)
                if projects:
                    selected_projects = _select_requested_projects(question, projects)
                    bullets = "\n".join(
                        f"- {name} — {description} [1]"
                        for name, description in selected_projects
                    )
                    return Answer(bullets, [result])

        if "education" in lowered_question or "eğitim" in lowered_question:
            for result in unique_results:
                section = _extract_section(
                    self.store.get_document_content(result.source),
                    "Education",
                    ("Projects", "Experience", "Skills"),
                )
                education = _format_education(section)
                if education:
                    return Answer(f"{education} [1]", [result])

        if "technolog" not in lowered_question:
            return None

        query_terms = _meaningful_terms(question)
        candidates: list[tuple[int, SearchResult, str]] = []
        for result in unique_results:
            content = self.store.get_document_content(result.source)
            for match in re.finditer(
                r"(?im)^\s*Technologies\s*:\s*([^\r\n]+)", content
            ):
                nearby = content[max(0, match.start() - 900) : match.start()]
                nearby_terms = _meaningful_terms(nearby)
                score = sum(
                    1
                    for term in query_terms
                    if any(_terms_are_close(term, item) for item in nearby_terms)
                )
                technologies = match.group(1).strip(" .")
                candidates.append((score, result, technologies))
        if not candidates:
            return None
        score, source, technologies = max(candidates, key=lambda item: item[0])
        if score < 2:
            return None
        return Answer(f"{technologies}. [1]", [source])

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
    technology_instruction = (
        " For technology questions, return only the technologies explicitly listed "
        "for the named project."
        if "technolog" in question.lower()
        else ""
    )
    return (
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question.strip()}\n\n"
        "IMPORTANT: Answer only in English. Use only facts explicitly supported by "
        "the context. For list questions, extract every item under the relevant section "
        "and return only short bullets; never stop after the first item. For project "
        "lists, include each project name followed by one short supported description. "
        f"Do not expand abbreviations or invent explanations.{technology_instruction}"
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


def _is_project_list_question(question: str) -> bool:
    lowered = question.lower()
    return "project" in lowered or "proje" in lowered


def _requested_profile_section(
    question: str,
) -> tuple[str, tuple[str, ...]] | None:
    terms = _meaningful_terms(question)
    all_headings = (
        "About Me",
        "Education",
        "Projects",
        "Experience",
        "Skills",
        "Certificates",
        "Sertificates",
    )
    if terms & {"experience", "experiences", "employment", "deneyim", "deneyimleri"}:
        return "Experience", tuple(item for item in all_headings if item != "Experience")
    if terms & {"skill", "skills", "yetenek", "yetenekler", "beceri", "beceriler"}:
        return "Skills", tuple(item for item in all_headings if item != "Skills")
    if terms & {"certificate", "certificates", "sertificate", "sertificates", "sertifika", "sertifikalar"}:
        # Some CVs contain the misspelled heading "Sertificates".
        return "Certificates", tuple(item for item in all_headings if item != "Certificates")
    return None


def _format_profile_section(heading: str, section: str) -> str:
    if not section:
        return ""
    lines = []
    for raw_line in section.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" \tâ€¢")
        if not line or line.lower() == heading.lower():
            continue
        if re.fullmatch(r"\[Sayfa\s+\d+]", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    if not lines:
        return ""
    return f"{heading}:\n" + "\n".join(f"- {line}" for line in lines)


def _extract_section(
    content: str, heading: str, following_headings: tuple[str, ...]
) -> str:
    start_match = re.search(
        rf"(?im)^\s*{re.escape(heading)}\s*$", content
    )
    if not start_match:
        return ""
    end = len(content)
    remainder = content[start_match.end() :]
    for candidate in following_headings:
        match = re.search(
            rf"(?im)^\s*{re.escape(candidate)}\s*$", remainder
        )
        if match:
            end = min(end, start_match.end() + match.start())
    return content[start_match.start() : end].strip()


def _extract_projects(section: str) -> list[tuple[str, str]]:
    if not section:
        return []
    lines = [
        re.sub(r"\s+", " ", line).strip(" \t•")
        for line in section.splitlines()
    ]
    lines = [line for line in lines if line and line.lower() != "projects"]
    role_words = ("developer", "designer", "manager", "engineer")
    projects: list[tuple[str, str]] = []
    for index in range(len(lines) - 1):
        name = lines[index]
        role = lines[index + 1].lower()
        if (
            not any(word in role for word in role_words)
            or name.lower().startswith(("technologies:", "kardelen tumay"))
            or name.startswith("•")
        ):
            continue
        description_parts: list[str] = []
        for line in lines[index + 2 :]:
            lowered = line.lower()
            if lowered.startswith("technologies:") or line.startswith("•"):
                break
            description_parts.append(line)
            if re.search(r"[.!?]$", line):
                break
        description = " ".join(description_parts)
        description = re.sub(r"(?<=[a-z])- (?=[a-z])", "", description)
        sentence = re.split(r"(?<=[.!?])\s+", description, maxsplit=1)[0]
        if sentence:
            projects.append((name, sentence))
    return projects


def _select_requested_projects(
    question: str, projects: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    query_terms = _meaningful_terms(question) - {
        "kardelen", "project", "projects", "proje", "projeler",
    }
    if not query_terms:
        return projects

    ranked: list[tuple[int, tuple[str, str]]] = []
    for project in projects:
        name_terms = _meaningful_terms(project[0])
        score = sum(
            1
            for query_term in query_terms
            if any(_terms_are_close(query_term, name_term) for name_term in name_terms)
        )
        ranked.append((score, project))

    best_score = max(score for score, _ in ranked)
    if best_score == 0:
        return projects
    return [project for score, project in ranked if score == best_score]


def _format_education(section: str) -> str:
    if not section:
        return ""
    lines = [
        re.sub(r"\s+", " ", line).strip(" \t•")
        for line in section.splitlines()
    ]
    lines = [line for line in lines if line and line.lower() != "education"]
    if len(lines) < 2:
        return ""
    institution = lines[0]
    program = lines[1]
    return f"{institution} — {program}."


def _is_person_identity_question(question: str) -> bool:
    lowered = question.lower().strip()
    return bool(
        re.search(r"\bwho\s+is\b", lowered)
        or re.search(r"\bkimdir\b", lowered)
        or re.search(r"\bkim\s+", lowered)
    )


def _person_profile_answer(question: str, content: str) -> str:
    section = _extract_section(
        content,
        "About Me",
        ("Education", "Projects", "Experience", "Skills"),
    )
    if not section:
        return ""

    before_section = content[: content.lower().find("about me")]
    name_candidates = [
        line.strip()
        for line in before_section.splitlines()
        if re.fullmatch(r"[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü'-]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü'-]+)+", line.strip())
    ]
    if not name_candidates:
        return ""
    name = name_candidates[-1]
    question_terms = _meaningful_terms(question)
    if not (_meaningful_terms(name) & question_terms):
        return ""

    profile = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", section)
    profile = re.sub(r"\s+", " ", profile).strip()
    profile = re.sub(r"^About Me\s*", "", profile, flags=re.IGNORECASE)
    first_sentence = re.split(r"(?<=[.!?])\s+", profile, maxsplit=1)[0].strip()
    if not first_sentence:
        return ""
    return f"{name} is a {first_sentence[0].lower() + first_sentence[1:]}"


def _extract_spreadsheet_summary(question: str, content: str) -> str:
    query_terms = _meaningful_terms(question) - {"many", "much"}
    if not query_terms:
        return ""

    candidates: list[tuple[int, float, str, str]] = []
    for line in content.splitlines():
        cells = re.findall(
            r"([A-Z]+)(\d+)=([^|]+?)(?=\s*\|\s*[A-Z]+\d+=|$)", line
        )
        for index, (column, row, raw_label) in enumerate(cells[:-1]):
            next_column, next_row, raw_value = cells[index + 1]
            if row != next_row or _column_number(next_column) != _column_number(column) + 1:
                continue
            label = raw_label.strip()
            value = raw_value.strip()
            if not _is_number(value):
                continue
            label_terms = _meaningful_terms(label)
            matched = sum(
                1
                for query_term in query_terms
                if any(_terms_are_close(query_term, label_term) for label_term in label_terms)
            )
            if matched:
                candidates.append((matched, matched / len(query_terms), label, value))

    if not candidates:
        return ""
    matched, coverage, label, value = max(
        candidates, key=lambda item: (item[0], item[1], len(item[2]))
    )
    if matched < 2 and coverage < 1.0:
        return ""
    return f"{label}: {_format_summary_number(label, value)}."


def _analyze_spreadsheet(question: str, content: str) -> str:
    lowered = question.lower()
    highest_requested = any(
        term in lowered
        for term in ("highest", "largest", "most", "maximum", "en yüksek", "en fazla")
    )
    total_requested = any(term in lowered for term in ("total", "sum", "toplam"))
    if not highest_requested and not total_requested:
        return ""

    tables = _spreadsheet_tables(content)
    query_terms = _meaningful_terms(question)
    for headers, rows in tables:
        metric = _best_matching_header(
            query_terms,
            headers,
            rows,
            numeric=True,
        )
        if not metric:
            continue

        if highest_requested:
            dimension = _best_matching_header(
                query_terms,
                headers,
                rows,
                numeric=False,
            )
            if not dimension:
                continue
            grouped: dict[str, float] = {}
            for row in rows:
                label = row.get(dimension, "").strip()
                value = row.get(metric, "").strip()
                if label and _is_number(value):
                    grouped[label] = grouped.get(label, 0.0) + float(value.replace(",", ""))
            if grouped:
                label, value = max(grouped.items(), key=lambda item: item[1])
                return f"{label} had the highest {metric}: {_format_metric_value(metric, value)}."

        if total_requested:
            filters: list[tuple[str, str]] = []
            metric_terms = _meaningful_terms(metric)
            for header in headers:
                if header == metric:
                    continue
                for value in {row.get(header, "").strip() for row in rows}:
                    value_terms = _meaningful_terms(value)
                    describes_metric = bool(value_terms) and all(
                        any(_terms_are_close(term, metric_term) for metric_term in metric_terms)
                        for term in value_terms - {"total", "toplam"}
                    )
                    if (
                        value
                        and not _is_number(value)
                        and value.lower() in lowered
                        and not describes_metric
                    ):
                        filters.append((header, value))
            matching_rows = [
                row
                for row in rows
                if all(row.get(header, "").strip() == value for header, value in filters)
            ]
            values = [
                float(row[metric].replace(",", ""))
                for row in matching_rows
                if _is_number(row.get(metric, ""))
            ]
            if values:
                if not filters:
                    return f"Total {metric}: {_format_metric_value(metric, sum(values))}."
                filter_text = ", ".join(value for _, value in filters)
                return f"Total {metric} for {filter_text}: {_format_metric_value(metric, sum(values))}."
    return ""


def _spreadsheet_tables(content: str) -> list[tuple[list[str], list[dict[str, str]]]]:
    sheet_rows: dict[str, dict[int, dict[str, str]]] = {}
    current_sheet = "Workbook"
    for line in content.splitlines():
        sheet_match = re.match(r"\[Çalışma Sayfası:\s*(.+?)]", line.strip())
        if sheet_match:
            current_sheet = sheet_match.group(1)
            sheet_rows.setdefault(current_sheet, {})
            continue
        for column, row_text, value in re.findall(
            r"([A-Z]+)(\d+)=([^|]+?)(?=\s*\|\s*[A-Z]+\d+=|$)", line
        ):
            row_number = int(row_text)
            sheet_rows.setdefault(current_sheet, {}).setdefault(row_number, {})[
                column
            ] = value.strip()

    tables: list[tuple[list[str], list[dict[str, str]]]] = []
    for rows_by_number in sheet_rows.values():
        ordered = sorted(rows_by_number.items())
        for header_index, (header_row_number, header_cells) in enumerate(ordered):
            if len(header_cells) < 2 or not all(
                value and not _is_number(value) for value in header_cells.values()
            ):
                continue
            columns = sorted(header_cells, key=_column_number)
            headers = [header_cells[column] for column in columns]
            data_rows: list[dict[str, str]] = []
            for row_number, cells in ordered[header_index + 1 :]:
                if row_number <= header_row_number:
                    continue
                mapped = {
                    header_cells[column]: cells.get(column, "")
                    for column in columns
                }
                if sum(bool(value) for value in mapped.values()) >= 2:
                    data_rows.append(mapped)
            if data_rows:
                tables.append((headers, data_rows))
                break
    return tables


def _best_matching_header(
    query_terms: set[str],
    headers: list[str],
    rows: list[dict[str, str]],
    *,
    numeric: bool,
) -> str:
    ranked: list[tuple[int, int, str]] = []
    for header in headers:
        values = [row.get(header, "") for row in rows if row.get(header, "")]
        if not values:
            continue
        numeric_ratio = sum(_is_number(value) for value in values) / len(values)
        if numeric != (numeric_ratio >= 0.7):
            continue
        header_terms = _meaningful_terms(header)
        score = sum(
            1
            for query_term in query_terms
            if any(_terms_are_close(query_term, header_term) for header_term in header_terms)
        )
        if score:
            extra_terms = len(
                {
                    term
                    for term in header_terms
                    if not any(_terms_are_close(term, query_term) for query_term in query_terms)
                }
            )
            ranked.append((score, -extra_terms, header))
    return max(ranked, default=(0, 0, ""), key=lambda item: (item[0], item[1]))[2]


def _format_metric_value(metric: str, value: float) -> str:
    formatted = f"{int(value):,}" if value.is_integer() else f"{value:,.2f}"
    return f"{formatted} TRY" if "try" in metric.lower() else formatted


def _column_number(column: str) -> int:
    number = 0
    for character in column:
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def _format_summary_number(label: str, value: str) -> str:
    number = float(value.replace(",", ""))
    if "margin" in label.lower() and abs(number) <= 1:
        return f"{number:.1%}"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def _has_named_anchor(question: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]+", question)
    capitalized = [token for token in tokens[1:] if token[0].isupper()]
    return len(capitalized) >= 2 or any(token.isupper() and len(token) >= 3 for token in capitalized)


def _needs_project_completion(text: str) -> bool:
    lowered = text.lower()
    return len(text.split()) < 18 or "http" not in lowered and "system" not in lowered


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
            if not clean:
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
                best = max(best, coverage)
            if (
                len(clean) > 80
                or re.search(r"[.!?]$", clean)
                or not clean[0].isupper()
                or "," in clean
            ):
                continue
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
        end_index = min(len(sentences), best_index + 30)
        for index in range(best_index + 1, end_index):
            if _looks_like_section_heading(sentences[index]):
                end_index = index
                break
        chosen_indices = set(range(best_index, end_index))
    else:
        chosen_indices = {index for _, index, _ in relevant[:3]}
    excerpt = " ".join(sentences[index] for index in sorted(chosen_indices))
    return excerpt[:max_chars]


def _looks_like_section_heading(sentence: str) -> bool:
    clean = sentence.strip(" \t•-*:#")
    words = clean.split()
    return (
        bool(clean)
        and len(clean) <= 40
        and len(words) <= 4
        and clean[0].isupper()
        and not re.search(r"[.!?,;:/()-]", clean)
    )


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
    allow_named_anchor: bool = False,
) -> bool:
    if not results:
        return False
    if allow_named_anchor:
        return True
    if max(item.score for item in results) >= MIN_ANSWER_RELEVANCE:
        return True

    query_terms = _meaningful_terms(question)
    if len(query_terms) == 1:
        term = next(iter(query_terms))
        evidence_terms = _meaningful_terms(
            " ".join(item.content for item in results)
        )
        return (
            any(_terms_are_close(term, evidence_term) for evidence_term in evidence_terms)
            and max(item.score for item in results) >= 0.25
        )
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
            and len(matched_terms) / len(query_terms) >= 0.2
        )
    if distinctive_matches >= 2 and len(matched_terms) / len(query_terms) >= 0.2:
        return True
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


def _extract_glossary_definition(content: str, term: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in content.splitlines()]
    normalized_term = term.casefold().strip()
    for index, line in enumerate(lines):
        if line.casefold() != normalized_term:
            continue
        definition_parts: list[str] = []
        for candidate in lines[index + 1 : index + 12]:
            if not candidate:
                continue
            if re.fullmatch(r"\[Sayfa\s+\d+]", candidate, flags=re.IGNORECASE):
                break
            definition_parts.append(candidate)
            if re.search(r"[.!?]$", candidate):
                break
        definition = " ".join(definition_parts).strip()
        if definition and re.search(r"[.!?]$", definition):
            return definition
    return ""


def _filter_comparison_results(
    facets: list[str], results: list[SearchResult]
) -> list[SearchResult]:
    facet_terms = [_meaningful_terms(facet) for facet in facets]
    focused: list[SearchResult] = []
    covered_facets: set[int] = set()
    for result in results:
        content_terms = _meaningful_terms(result.content)
        matched_facets = {
            index
            for index, terms in enumerate(facet_terms)
            if terms and all(
                any(_terms_are_close(term, content_term) for content_term in content_terms)
                for term in terms
            )
        }
        if matched_facets:
            focused.append(result)
            covered_facets.update(matched_facets)
    return focused if len(covered_facets) == len(facet_terms) else []


def _has_comparison_evidence(
    facets: list[str], results: list[SearchResult]
) -> bool:
    if not facets or not results:
        return False
    evidence_terms = _meaningful_terms(" ".join(item.content for item in results))
    return all(
        terms
        and all(
            any(_terms_are_close(term, evidence_term) for evidence_term in evidence_terms)
            for term in terms
        )
        for terms in (_meaningful_terms(facet) for facet in facets)
    )


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
