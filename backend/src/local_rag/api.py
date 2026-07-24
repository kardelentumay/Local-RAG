from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .foundry import FoundryChat, FoundryEmbeddings, FoundryRuntime
from .models import Answer
from .service import RAGService
from .store import SQLiteStore


LOGGER = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/knowledge.db")
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DEFAULT_CHAT_MODEL = "qwen2.5-1.5b"
DEFAULT_DOCUMENTS_PATH = Path("documents")
ALLOWED_DOCUMENT_SUFFIXES = {".pdf", ".md", ".txt"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


class HealthResponse(BaseModel):
    status: str
    mode: str
    models_loaded: bool


class StatusResponse(BaseModel):
    database: str
    documents: int
    chunks: int
    models_loaded: bool


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=4000)]
    top_k: Annotated[int, Field(ge=1, le=5)] = 2


class SourceResponse(BaseModel):
    number: int
    document: str
    chunk: int
    score: float
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class DocumentResponse(BaseModel):
    source: str
    name: str
    kind: str
    category: str
    chunks: int
    updated_at: str


class LocalRAGRuntime:
    """Owns the local model lifecycle and serializes Foundry inference calls."""

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_model: str | None = None,
        chat_model: str | None = None,
        documents_path: Path | None = None,
    ):
        configured_db = os.environ.get("LOCAL_RAG_DB")
        self.db_path = db_path or (Path(configured_db) if configured_db else DEFAULT_DB_PATH)
        configured_documents = os.environ.get("LOCAL_RAG_DOCUMENTS")
        self.documents_path = documents_path or (
            Path(configured_documents) if configured_documents else DEFAULT_DOCUMENTS_PATH
        )
        self.embedding_model = embedding_model or os.environ.get(
            "LOCAL_RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.chat_model = chat_model or os.environ.get("LOCAL_RAG_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self._foundry: FoundryRuntime | None = None
        self._service: RAGService | None = None
        self._lock = threading.Lock()

    @property
    def models_loaded(self) -> bool:
        return self._service is not None

    def stats(self) -> tuple[int, int]:
        return SQLiteStore(self.db_path).stats()

    def ask(self, question: str, top_k: int) -> Answer:
        with self._lock:
            service = self._get_service()
            return service.answer(question, top_k)

    def list_documents(self) -> list[dict[str, object]]:
        return SQLiteStore(self.db_path).list_documents()

    def ingest_document(self, path: Path, category: str) -> None:
        with self._lock:
            self._get_service().ingest(
                path,
                source_root=self.documents_path,
                category=category,
            )

    def delete_document(self, source: str) -> bool:
        with self._lock:
            return SQLiteStore(self.db_path).delete_document(source)

    def close(self) -> None:
        with self._lock:
            if self._foundry is not None:
                self._foundry.close()
            self._service = None
            self._foundry = None

    def _get_service(self) -> RAGService:
        if self._service is None:
            foundry = FoundryRuntime()
            try:
                embeddings = FoundryEmbeddings(foundry, self.embedding_model)
                chat = FoundryChat(foundry, self.chat_model)
            except Exception:
                foundry.close()
                raise
            self._foundry = foundry
            self._service = RAGService(SQLiteStore(self.db_path), embeddings, chat)
        return self._service


runtime = LocalRAGRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await asyncio.to_thread(runtime.close)


app = FastAPI(
    title="Local RAG API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", mode="local", models_loaded=runtime.models_loaded)


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    try:
        documents, chunks = await asyncio.to_thread(runtime.stats)
    except Exception as exc:
        LOGGER.exception("Could not read the local RAG database status.")
        raise HTTPException(status_code=500, detail="The local database is unavailable.") from exc
    return StatusResponse(
        database=str(runtime.db_path),
        documents=documents,
        chunks=chunks,
        models_loaded=runtime.models_loaded,
    )


def _document_response(item: dict[str, object]) -> DocumentResponse:
    source = str(item["source"])
    name = Path(source).name
    if source.startswith("uploads/"):
        name = re.sub(r"^[0-9a-f]{32}-", "", name)
    suffix = Path(source).suffix.lower().lstrip(".")
    return DocumentResponse(
        source=source,
        name=name,
        kind=suffix.upper() or "FILE",
        category=str(item["category"]),
        chunks=int(item["chunks"]),
        updated_at=str(item["updated_at"]),
    )


@app.get("/api/documents", response_model=list[DocumentResponse])
async def documents() -> list[DocumentResponse]:
    try:
        items = await asyncio.to_thread(runtime.list_documents)
    except Exception as exc:
        LOGGER.exception("Could not list indexed documents.")
        raise HTTPException(status_code=500, detail="Documents could not be listed.") from exc
    return [_document_response(item) for item in items]


@app.post("/api/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(..., min_length=1, max_length=80),
) -> DocumentResponse:
    clean_category = _normalize_category(category)
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_DOCUMENT_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Only PDF, Markdown, and plain-text files are supported.",
        )
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip(".-_")
    safe_name = f"{safe_stem or 'document'}{suffix}"
    upload_root = (runtime.documents_path / "uploads").resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    target = (upload_root / f"{uuid.uuid4().hex}-{safe_name}").resolve()
    if upload_root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    size = 0
    try:
        with target.open("xb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="The file exceeds the 25 MB upload limit.",
                    )
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        await asyncio.to_thread(runtime.ingest_document, target, clean_category)
        items = await asyncio.to_thread(runtime.list_documents)
        source = target.relative_to(runtime.documents_path.resolve()).as_posix()
        item = next(entry for entry in items if entry["source"] == source)
        return _document_response(item)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        LOGGER.exception("Could not ingest uploaded document.")
        raise HTTPException(status_code=500, detail="The document could not be indexed.") from exc
    finally:
        await file.close()


def _normalize_category(category: str) -> str:
    clean = " ".join(category.split())
    if not clean or any(ord(character) < 32 for character in clean):
        raise HTTPException(status_code=422, detail="Category name is invalid.")
    return clean


@app.delete("/api/documents/{source:path}", status_code=204)
async def delete_document(source: str) -> None:
    clean_source = Path(source).as_posix().lstrip("/")
    if not clean_source or ".." in Path(clean_source).parts:
        raise HTTPException(status_code=400, detail="Invalid document path.")
    deleted = await asyncio.to_thread(runtime.delete_document, clean_source)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    if clean_source.startswith("uploads/"):
        upload_root = (runtime.documents_path / "uploads").resolve()
        target = (runtime.documents_path / clean_source).resolve()
        if upload_root in target.parents:
            target.unlink(missing_ok=True)


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")
    try:
        answer = await asyncio.to_thread(runtime.ask, question, request.top_k)
    except (RuntimeError, ValueError) as exc:
        LOGGER.exception("Local model inference failed.")
        raise HTTPException(
            status_code=503,
            detail="The local model service could not complete the request.",
        ) from exc
    sources = [
        SourceResponse(
            number=index,
            document=item.source,
            chunk=item.position + 1,
            score=round(item.score, 4),
            content=item.content,
        )
        for index, item in enumerate(answer.sources, start=1)
    ]
    return AskResponse(answer=answer.text, sources=sources)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "local_rag.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        access_log=False,
    )
