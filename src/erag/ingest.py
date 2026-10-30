"""Ingestion: parse -> scan for poisoned content -> section-aware chunking.

Chunks never cross section boundaries, carry full ACL metadata, and overlap by whole
sentences so a fact is not cut in half by a boundary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Chunk, Classification, Document
from .textutil import split_sentences

# Heuristic scanner for indirect prompt injection planted inside documents.
# It is a tripwire, not a guarantee; the real defences are ACL enforcement outside the model,
# treating sources as data in the prompt, and citation verification on the output.
_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|rules|context)"
    r"|reveal\s+(the\s+)?(system\s+prompt|confidential|restricted)"
    r"|you\s+are\s+now\s+(in\s+)?(developer|admin|dan)\s+mode"
    r"|(print|output|dump)\s+(all|every)\s+(restricted|confidential|secret)",
    re.I,
)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    quarantined: list[str] = field(default_factory=list)  # doc_ids held back


def _sections(doc: Document) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    current, buf = "Overview", []
    for line in doc.text.splitlines():
        if line.startswith("## "):
            if buf:
                out.append((current, " ".join(buf).strip()))
            current, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        elif line.strip():
            buf.append(line.strip())
    if buf:
        out.append((current, " ".join(buf).strip()))
    return out


def _window(items: list[tuple[str, str]], doc: Document, max_words: int, overlap: int, pos: int) -> tuple[list[Chunk], int]:
    """Greedy sentence packing over (section, sentence) items with sentence-level overlap."""
    out: list[Chunk] = []
    window: list[tuple[str, str]] = []
    words = 0
    for sec, s in items:
        w = len(s.split())
        if window and words + w > max_words:
            out.append(_mk(doc, window, pos)); pos += 1
            window = window[-overlap:] if overlap else []
            words = sum(len(x[1].split()) for x in window)
            if window and words + w > max_words:
                window, words = [], 0
        window.append((sec, s)); words += w
    if window:
        out.append(_mk(doc, window, pos)); pos += 1
    return out, pos


def chunk_document(doc: Document, max_words: int = 120, overlap_sentences: int = 1, pack: bool = False) -> list[Chunk]:
    """pack=False: chunks never cross section boundaries (split only if a section exceeds max_words).
    pack=True : adjacent sections are merged until max_words (better for many tiny sections)."""
    secs = [(name, [(name, s) for s in split_sentences(body)]) for name, body in _sections(doc)]
    chunks: list[Chunk] = []
    pos = 0
    if pack:
        items = [it for _, its in secs for it in its]
        chunks, pos = _window(items, doc, max_words, overlap_sentences, pos)
    else:
        for _, its in secs:
            cs, pos = _window(its, doc, max_words, overlap_sentences, pos)
            chunks += cs
    return chunks


def _mk(doc: Document, window: list[tuple[str, str]], pos: int) -> Chunk:
    sections = list(dict.fromkeys(sec for sec, _ in window))
    return Chunk(chunk_id=f"{doc.doc_id}#{pos}", doc_id=doc.doc_id, title=doc.title, section=" / ".join(sections),
                 text=" ".join(s for _, s in window), department=doc.department,
                 classification=doc.classification, allowed_groups=doc.allowed_groups, position=pos)


def ingest(docs: list[Document], max_words: int = 120, overlap_sentences: int = 1,
           quarantine_injections: bool = True, pack: bool = False) -> tuple[list[Chunk], IngestReport]:
    report = IngestReport()
    chunks: list[Chunk] = []
    for d in docs:
        if quarantine_injections and looks_like_injection(d.text):
            report.quarantined.append(d.doc_id)
            continue
        cs = chunk_document(d, max_words, overlap_sentences, pack)
        chunks.extend(cs)
        report.documents += 1
    report.chunks = len(chunks)
    return chunks, report


def load_documents(path: str | Path) -> list[Document]:
    docs = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        docs.append(Document(doc_id=r["doc_id"], title=r["title"], text=r["text"], department=r["department"],
                             classification=Classification[r["classification"]],
                             allowed_groups=tuple(r.get("allowed_groups", [])), source=r.get("source", ""),
                             version=r.get("version", "1.0")))
    return docs


def save_documents(docs: list[Document], path: str | Path) -> None:
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps({"doc_id": d.doc_id, "title": d.title, "text": d.text, "department": d.department,
                                "classification": d.classification.name, "allowed_groups": list(d.allowed_groups),
                                "source": d.source, "version": d.version}) + "\n")
