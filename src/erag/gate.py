"""Answerability gate: decide whether the retrieved context can support an answer.

confidence = IDF-weighted query-term coverage of the best retrieved chunks
             x 0.5 for every *named entity* in the question that appears in none of them.

The entity term catches the dangerous failure for enterprise RAG: asking about X (a vendor, region,
service, level) and getting a confident answer about its sibling Y because the documents look alike.
"""
from __future__ import annotations

import re

from .textutil import tokenize

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")


def entities(question: str) -> list[list[str]]:
    """Heuristic named-entity spotter: Title-case words after the first word, hyphenated identifiers,
    and letter+digit codes (L12, SEV1). ALL-CAPS acronyms (SLA, MFA) are deliberately ignored: users
    use them as generic vocabulary. Returns each entity as a list of normalised tokens."""
    out: list[list[str]] = []
    for i, w in enumerate(_WORD.findall(question)):
        title = i > 0 and w[0].isupper() and w[1:].islower()
        hyphen = "-" in w
        code = bool(re.fullmatch(r"[A-Za-z]+\d+", w)) and not w.isalpha()
        if title or hyphen or code:
            toks = tokenize(w.replace("-", " "))
            if toks:
                out.append(toks)
    return out


def confidence(retriever, question: str, hits, top: int = 3, use_entities: bool = True) -> tuple[float, int, float]:
    """Returns (confidence, missing_entities, raw_coverage)."""
    q_tokens = tokenize(question)
    top_hits = hits[:top]
    cov = max((retriever.coverage(q_tokens, h.idx) for h in top_hits), default=0.0)
    present: set[str] = set()
    for h in top_hits:
        present |= retriever._terms[h.idx]
    missing = sum(1 for ent in entities(question) if not all(t in present for t in ent)) if use_entities else 0
    return cov * (0.5 ** missing), missing, cov
