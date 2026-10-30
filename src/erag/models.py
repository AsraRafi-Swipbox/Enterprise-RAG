"""Core data model. Everything the ACL layer needs travels with the chunk."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Classification(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str  # markdown-ish: "# Title" then "## Section" blocks
    department: str
    classification: Classification
    allowed_groups: tuple[str, ...] = ()
    source: str = ""
    version: str = "1.0"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    department: str
    classification: Classification
    allowed_groups: tuple[str, ...]
    position: int

    @property
    def indexed_text(self) -> str:
        """Contextual header + body. The header makes 'EMEA' in the title searchable
        from a chunk that only says 'the nightly cap is 180 EUR'."""
        return f"{self.title}. {self.section}. {self.text}"


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    clearance: Classification
    departments: frozenset[str] = frozenset()
    groups: frozenset[str] = frozenset()


@dataclass
class Citation:
    n: int
    doc_id: str
    chunk_id: str
    title: str
    section: str
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    abstained: bool
    confidence: float
    request_id: str = ""
    latency_ms: float = 0.0
    trace: dict = field(default_factory=dict)
