from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    position: int
    content: str


@dataclass(frozen=True)
class SearchResult:
    source: str
    position: int
    content: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchResult]
