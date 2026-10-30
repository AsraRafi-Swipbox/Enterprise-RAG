"""Access-control policy (RBAC clearance + ABAC need-to-know) and fast per-query masks.

Rule for reading a document/chunk:
  1. user.clearance >= doc.classification            (clearance ceiling)
  2. if doc is PUBLIC/INTERNAL                        -> allowed
  3. else need-to-know: doc.department in user.departments
     OR doc.allowed_groups intersects user.groups     (explicit grant)
Deny by default. The decision is taken from the *directory*, never from the LLM or the
request body.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import Chunk, Classification, User


def can_read(user: User, classification: Classification, department: str,
             allowed_groups: Iterable[str]) -> bool:
    if user.clearance < classification:
        return False
    if classification <= Classification.INTERNAL:
        return True
    if department in user.departments:
        return True
    return bool(set(allowed_groups) & user.groups)


class AccessMask:
    """Pre-computes one ACL 'signature' per chunk so a user mask is a single array lookup.

    There are only a handful of distinct (classification, dept, groups) signatures in any
    corpus, so evaluating the policy once per signature makes mask construction O(N) with a
    tiny constant (one fancy-index) instead of N Python policy calls per query.
    """

    def __init__(self, chunks: list[Chunk]):
        sigs: dict[tuple, int] = {}
        ids = np.empty(len(chunks), dtype=np.int32)
        for i, c in enumerate(chunks):
            sig = (c.classification, c.department, tuple(sorted(c.allowed_groups)))
            ids[i] = sigs.setdefault(sig, len(sigs))
        self._sig_ids = ids
        self._sigs = list(sigs)
        self._cache: dict[tuple, np.ndarray] = {}

    def for_user(self, user: User) -> np.ndarray:
        key = (user.clearance, user.departments, user.groups)
        if key not in self._cache:
            allowed = np.array([can_read(user, c, d, g) for (c, d, g) in self._sigs], dtype=bool)
            self._cache[key] = allowed[self._sig_ids]
        return self._cache[key]


class UserDirectory:
    """Source of truth for who a caller *is*. Tokens only carry the subject id."""

    def __init__(self, users: Iterable[User]):
        self._users = {u.user_id: u for u in users}

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def all(self) -> list[User]:
        return list(self._users.values())

    @classmethod
    def from_json(cls, path: str | Path) -> "UserDirectory":
        raw = json.loads(Path(path).read_text())
        return cls(User(
            user_id=r["user_id"], name=r["name"], clearance=Classification[r["clearance"]],
            departments=frozenset(r.get("departments", [])), groups=frozenset(r.get("groups", [])),
        ) for r in raw)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([{
            "user_id": u.user_id, "name": u.name, "clearance": u.clearance.name,
            "departments": sorted(u.departments), "groups": sorted(u.groups)}
            for u in self._users.values()], indent=2))
