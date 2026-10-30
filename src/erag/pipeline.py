"""End-to-end orchestration: authorize -> mask -> retrieve -> answerability gate -> generate ->
verify citations -> audit."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import numpy as np

from .access import AccessMask
from .audit import AuditLog
from .gate import confidence as gate_confidence
from .generation import ExtractiveGenerator, Generator, verify_and_build
from .models import Answer, Chunk, User
from .retrieval import HybridRetriever

ABSTAIN_TEXT = "I couldn't find a reliable answer in the documents you have access to."


@dataclass
class PipelineConfig:
    top_k: int = 5
    mode: str = "hybrid+rerank"
    abstain_threshold: float = 0.55   # calibrated on a dev split (see scripts/run_eval.py)
    min_support: float = 0.6
    entity_gate: bool = True


class RAGPipeline:
    def __init__(self, chunks: list[Chunk], retriever: HybridRetriever | None = None,
                 generator: Generator | None = None, config: PipelineConfig | None = None,
                 audit: AuditLog | None = None):
        self.chunks = chunks
        self.cfg = config or PipelineConfig()
        self.retriever = retriever or HybridRetriever(chunks)
        self.generator = generator or ExtractiveGenerator(embedder=self.retriever.embedder)
        self.acl = AccessMask(chunks)
        self.chunk_index = {c.chunk_id: i for i, c in enumerate(chunks)}
        self.audit = audit or AuditLog(None)

    def ask(self, user: User, question: str) -> Answer:
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        trace: dict = {}

        mask = self.acl.for_user(user)                      # 1. who may see what (from directory)
        trace["mask_ms"] = (time.perf_counter() - t0) * 1e3
        hits = self.retriever.retrieve(question, mask, self.cfg.top_k, self.cfg.mode, trace)  # 2. masked retrieval
        assert all(mask[h.idx] for h in hits), "ACL invariant violated"  # defence in depth

        conf, missing, cov = gate_confidence(self.retriever, question, hits, use_entities=self.cfg.entity_gate)
        trace.update(n_hits=len(hits), coverage=cov, missing_entities=missing)

        if not hits or conf < self.cfg.abstain_threshold:   # 3. answerability gate
            return self._finish(user, question, rid, t0, trace, Answer(ABSTAIN_TEXT, [], True, conf), mask)

        t1 = time.perf_counter()
        idf = self.retriever.bm25.idf_of
        draft = self.generator.draft(question, hits, idf)   # 4. generate
        if draft.not_found:
            return self._finish(user, question, rid, t0, trace, Answer(ABSTAIN_TEXT, [], True, conf), mask)
        text, cites, dropped = verify_and_build(draft, hits, idf, self.cfg.min_support)  # 5. verify
        trace["generate_ms"] = (time.perf_counter() - t1) * 1e3
        trace["dropped_unsupported"] = dropped
        if not cites:
            return self._finish(user, question, rid, t0, trace, Answer(ABSTAIN_TEXT, [], True, conf), mask)
        return self._finish(user, question, rid, t0, trace, Answer(text, cites, False, conf), mask)

    def _finish(self, user, question, rid, t0, trace, ans: Answer, mask) -> Answer:
        ans.request_id = rid
        ans.latency_ms = (time.perf_counter() - t0) * 1e3
        ans.trace = trace
        self.audit.write(request_id=rid, user_id=user.user_id, query=question, abstained=ans.abstained,
                         confidence=round(ans.confidence, 3), cited_docs=[c.doc_id for c in ans.citations],
                         visible_chunk_fraction=round(float(mask.mean()), 3), latency_ms=round(ans.latency_ms, 2))
        return ans

    def resolve_chunk(self, user: User, chunk_id: str) -> Chunk | None:
        """Citation drill-down. Unauthorised and non-existent both return None (-> 404)."""
        i = self.chunk_index.get(chunk_id)
        return self.chunks[i] if i is not None and self.acl.for_user(user)[i] else None
