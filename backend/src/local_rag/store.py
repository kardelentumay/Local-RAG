from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Sequence

from .models import Chunk, SearchResult


class SQLiteStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as db, db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'RAG Research',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    source TEXT NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    UNIQUE(source, position)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    source UNINDEXED,
                    position UNINDEXED,
                    content,
                    tokenize = 'unicode61'
                );
                """
            )
            document_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "category" not in document_columns:
                db.execute(
                    "ALTER TABLE documents ADD COLUMN category TEXT NOT NULL "
                    "DEFAULT 'RAG Research'"
                )
            chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            fts_count = db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            if chunk_count != fts_count:
                db.execute("DELETE FROM chunks_fts")
                db.execute(
                    "INSERT INTO chunks_fts(source, position, content) "
                    "SELECT source, position, content FROM chunks"
                )

    def has_document(self, source: str, content_hash: str) -> bool:
        with closing(self._connect()) as db:
            row = db.execute("SELECT 1 FROM documents WHERE source = ? AND content_hash = ?", (source, content_hash)).fetchone()
        return row is not None

    def replace_document(
        self,
        source: str,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        category: str = "RAG Research",
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Parça ve embedding sayıları eşit değil")
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM chunks_fts WHERE source = ?", (source,))
            db.execute("DELETE FROM documents WHERE source = ?", (source,))
            db.execute(
                "INSERT INTO documents(source, content_hash, category) VALUES (?, ?, ?)",
                (source, content_hash, category),
            )
            db.executemany(
                "INSERT INTO chunks(source, position, content, embedding) VALUES (?, ?, ?, ?)",
                [(item.source, item.position, item.content, json.dumps(list(vector))) for item, vector in zip(chunks, embeddings)],
            )
            db.executemany(
                "INSERT INTO chunks_fts(source, position, content) VALUES (?, ?, ?)",
                [(item.source, item.position, item.content) for item in chunks],
            )

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
        sources: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k en az 1 olmalıdır")
        with closing(self._connect()) as db:
            if sources:
                placeholders = ",".join("?" for _ in sources)
                rows = db.execute(
                    "SELECT source, position, content, embedding FROM chunks "
                    f"WHERE source IN ({placeholders})",
                    tuple(sources),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT source, position, content, embedding FROM chunks"
                ).fetchall()
        scored = []
        for row in rows:
            score = cosine_similarity(query_embedding, json.loads(row["embedding"]))
            scored.append(SearchResult(row["source"], row["position"], row["content"], score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def stats(self) -> tuple[int, int]:
        with closing(self._connect()) as db:
            documents = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return documents, chunks

    def list_documents(self) -> list[dict[str, object]]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT documents.source, documents.category, documents.updated_at, "
                "COUNT(chunks.id) AS chunks "
                "FROM documents LEFT JOIN chunks ON chunks.source = documents.source "
                "GROUP BY documents.source, documents.category, documents.updated_at "
                "ORDER BY documents.category, documents.updated_at DESC, documents.source"
            ).fetchall()
        return [
            {
                "source": row["source"],
                "category": row["category"],
                "updated_at": row["updated_at"],
                "chunks": int(row["chunks"]),
            }
            for row in rows
        ]

    def delete_document(self, source: str) -> bool:
        with closing(self._connect()) as db, db:
            exists = db.execute(
                "SELECT 1 FROM documents WHERE source = ?", (source,)
            ).fetchone()
            if exists is None:
                return False
            db.execute("DELETE FROM chunks_fts WHERE source = ?", (source,))
            db.execute("DELETE FROM documents WHERE source = ?", (source,))
        return True

    def keyword_search(
        self,
        query: str,
        limit: int = 20,
        sources: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        terms = _fts_terms(query)
        if not terms or limit < 1:
            return []
        expression = " OR ".join(f'"{term}"*' for term in terms)
        with closing(self._connect()) as db:
            if sources:
                placeholders = ",".join("?" for _ in sources)
                rows = db.execute(
                    "SELECT source, position, content, bm25(chunks_fts) AS rank "
                    "FROM chunks_fts WHERE chunks_fts MATCH ? "
                    f"AND source IN ({placeholders}) ORDER BY rank LIMIT ?",
                    (expression, *sources, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT source, position, content, bm25(chunks_fts) AS rank "
                    "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expression, limit),
                ).fetchall()
        return [
            SearchResult(row["source"], int(row["position"]), row["content"], -float(row["rank"]))
            for row in rows
        ]

    def get_window(self, source: str, position: int, before: int = 1, after: int = 0) -> str:
        start = max(0, position - before)
        end = position + after
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT content FROM chunks WHERE source = ? AND position BETWEEN ? AND ? ORDER BY position",
                (source, start, end),
            ).fetchall()
        return "\n".join(row["content"] for row in rows)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding boyutları eşleşmiyor; veritabanını yeniden oluşturun")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _fts_terms(query: str) -> list[str]:
    import re

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "what", "when", "where", "which",
        "who", "why", "with", "can", "does", "do", "systems", "system",
    }
    terms = []
    for token in re.findall(r"[^\W_]+", query.lower(), flags=re.UNICODE):
        if len(token) >= 3 and token not in stopwords and token not in terms:
            terms.append(token)
    return terms[:12]
