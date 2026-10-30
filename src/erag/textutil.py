"""Tokenisation shared by BM25, the embedder, the reranker and the answer-span selector."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

STOP = frozenset("""a an the of to in on for and or is are was were be been by with at as from that this it its
what which who whom how when where why do does did can could should would will may might must i we you they my
our your their me us them per after before about into through during between both either more most also only
own same so too very s t just there here if not no any all each every than then""".split())


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
        return w[:-1] if len(w) > 2 and w[-1] == w[-2] else w
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def numbers_in(text: str) -> set[str]:
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
