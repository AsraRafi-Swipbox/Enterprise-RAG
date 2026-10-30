"""Evaluation harness: retrieval quality, ACL enforcement, end-to-end answers, verifier fuzzing."""
from __future__ import annotations

import random
import statistics
import time
from collections import defaultdict

import numpy as np

from .access import AccessMask
from .generation import Draft, sentence_supported, verify_and_build
from .ingest import ingest
from .metrics import bootstrap_ci, hit_at_k, mrr, ndcg_at_k, value_in
from .models import Chunk, User
from .pipeline import PipelineConfig, RAGPipeline
from .retrieval import HybridRetriever, Hit

STYLES = ["direct", "paraphrase", "keyword"]


class GoldIndex:
    def __init__(self, chunks: list[Chunk]):
        self.by_doc: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            self.by_doc[c.doc_id].append(i)
        self.chunks = chunks

    def relevant(self, row: dict) -> set[int]:
        return {i for i in self.by_doc.get(row["gold_doc"], []) if row["gold_sentence"] in self.chunks[i].text}


# -------------------------------------------------------------------------- retrieval
def eval_retrieval(retr: HybridRetriever, acl: AccessMask, gold: GoldIndex, users: dict[str, User],
                   rows: list[dict], mode: str) -> dict:
    per = defaultdict(list)
    by_style = defaultdict(list)
    for r in rows:
        mask = acl.for_user(users[r["user_id"]])
        hits = retr.retrieve(r["question"], mask, 10, mode)
        rel = gold.relevant(r)
        rels = [h.idx in rel for h in hits]
        for k in (1, 3, 5, 10):
            per[f"recall@{k}"].append(hit_at_k(rels, k))
        per["mrr"].append(mrr(rels))
        drels = [h.chunk.doc_id == r["gold_doc"] for h in hits]
        per["doc_recall@1"].append(hit_at_k(drels, 1))
        per["doc_recall@5"].append(hit_at_k(drels, 5))
        per["ndcg@10"].append(ndcg_at_k(rels, 10, len(rel)))
        by_style[STYLES[r["style"]]].append(hit_at_k(rels, 5))
    out = {m: float(np.mean(v)) for m, v in per.items()}
    out["recall@5_ci"] = bootstrap_ci(per["recall@5"])
    out["mrr_ci"] = bootstrap_ci(per["mrr"])
    out["recall@5_by_style"] = {s: float(np.mean(v)) for s, v in by_style.items()}
    out["n"] = len(rows)
    return out


# -------------------------------------------------------------------------------- ACL
def eval_acl_strategies(retr: HybridRetriever, acl: AccessMask, gold: GoldIndex, users: dict[str, User],
                        rows: list[dict], k: int = 5) -> dict:
    """no_acl: search everything, return everything.  post_filter: search everything, drop unauthorised
    from the top-k afterwards.  pre_filter (this system): mask inside the rankers before top-k."""
    import re
    res = {s: dict(exposed=0, gold_forbidden=0, rec=[], broad_rel=[], broad_underfilled=0, n=0, nb=0)
           for s in ("no_acl", "post_filter", "pre_filter")}
    for r in rows:
        u = users[r["user_id"]]
        mask = acl.for_user(u)
        rel = gold.relevant(r) if r["kind"] == "answerable" else set()
        rx = re.compile(r["tag"]) if r["kind"] == "broad" else None
        for strat in res:
            if strat == "pre_filter":
                hits = retr.retrieve(r["question"], mask, k)
            else:
                hits = retr.retrieve(r["question"], None, k)
                if strat == "post_filter":
                    hits = [h for h in hits if mask[h.idx]]
            R = res[strat]; R["n"] += 1
            R["exposed"] += any(not mask[h.idx] for h in hits)
            if r["kind"] == "forbidden":
                R["gold_forbidden"] += any(h.chunk.doc_id == r["gold_doc"] for h in hits)
            if r["kind"] == "answerable":
                R["rec"].append(hit_at_k([h.idx in rel for h in hits], k))
            if rx is not None:
                n_rel = sum(1 for h in hits if mask[h.idx] and rx.search(h.chunk.text))
                R["broad_rel"].append(n_rel); R["nb"] += 1
                R["broad_underfilled"] += n_rel < k
    out = {}
    for s_, R in res.items():
        out[s_] = dict(unauthorised_exposure_rate=R["exposed"] / R["n"], exposed_queries=R["exposed"],
                       total_queries=R["n"], forbidden_gold_exposure=R["gold_forbidden"],
                       answerable_recall_at_k=float(np.mean(R["rec"])),
                       broad_relevant_results_at_k=float(np.mean(R["broad_rel"])),
                       broad_underfilled_rate=R["broad_underfilled"] / max(1, R["nb"]))
    return out


# ------------------------------------------------------------------------ end-to-end
def run_e2e(pipe: RAGPipeline, gold: GoldIndex, users: dict[str, User], rows: list[dict]) -> list[dict]:
    recs = []
    for r in rows:
        u = users[r["user_id"]]
        ans = pipe.ask(u, r["question"])
        mask = pipe.acl.for_user(u)
        cited_idx = [pipe.chunk_index[c.chunk_id] for c in ans.citations]
        rec = dict(r, conf=ans.confidence, abstained0=ans.abstained, answer=ans.text, latency_ms=ans.latency_ms,
                   trace=ans.trace, unauth_cited=any(not mask[i] for i in cited_idx))
        if r["gold_doc"] and r["kind"] == "answerable":
            rel = gold.relevant(r)
            rec["value_ok"] = value_in(ans.text, r["gold_value"])
            rec["cite_ok"] = any(i in rel for i in cited_idx)
        recs.append(rec)
    return recs


def score_at(recs: list[dict], t: float) -> dict:
    ans_ = [r for r in recs if r["kind"] == "answerable"]
    answered = lambda r: (not r["abstained0"]) and r["conf"] >= t
    ok = [r for r in ans_ if answered(r) and r["value_ok"] and r["cite_ok"]]
    out = dict(
        threshold=t, n_answerable=len(ans_),
        grounded_correct=len(ok) / len(ans_),
        answered_rate=sum(answered(r) for r in ans_) / len(ans_),
        false_refusal=sum(not answered(r) for r in ans_) / len(ans_),
        value_match=sum(answered(r) and r["value_ok"] for r in ans_) / len(ans_),
        citation_precision=(sum(r["cite_ok"] for r in ans_ if answered(r)) / max(1, sum(answered(r) for r in ans_))),
    )
    for kind in ("forbidden", "unanswerable_generic", "unanswerable_near", "adversarial"):
        g = [r for r in recs if r["kind"] == kind]
        if g:
            out[f"abstain_{kind}"] = sum(not answered(r) for r in g) / len(g)
    neg = [r for r in recs if r["kind"] in ("forbidden", "unanswerable_generic", "unanswerable_near")]
    out["abstain_negatives"] = sum(not answered(r) for r in neg) / max(1, len(neg))
    out["balanced"] = 0.5 * out["grounded_correct"] + 0.5 * out["abstain_negatives"]
    return out


def calibrate(dev: list[dict], grid=None) -> float:
    grid = grid if grid is not None else [round(x, 2) for x in np.arange(0.30, 0.96, 0.01)]
    scores = [(score_at(dev, t)["balanced"], t) for t in grid]
    best = max(s for s, _ in scores)
    best_ts = [t for s, t in scores if s >= best - 1e-9]
    return float(np.median(best_ts))


# ------------------------------------------------------------------- verifier fuzzing
def eval_verifier(pipe: RAGPipeline, gold: GoldIndex, rows: list[dict], seed: int = 3) -> dict:
    """Simulates LLM failure modes and measures how many the citation verifier blocks."""
    rng = random.Random(seed)
    idf = pipe.retriever.bm25.idf_of
    ans = [r for r in rows if r["kind"] == "answerable"]
    caught = dict(wrong_number=0, wrong_claim=0)
    passed_clean = 0
    n = 0
    for r in ans:
        rel = list(gold.relevant(r))
        if not rel:
            continue
        ch = pipe.chunks[rel[0]]
        sent = r["gold_sentence"]
        n += 1
        passed_clean += sentence_supported(sent, ch.text, idf)
        # (1) hallucinated number: change a digit string to one absent from the source
        import re
        m = re.search(r"\d[\d,]*", sent)
        if m:
            fake = str(int(m.group().replace(",", "")) * 7 + 13)
            caught["wrong_number"] += not sentence_supported(sent.replace(m.group(), fake, 1), ch.text, idf)
        else:
            caught["wrong_number"] += 1  # no numeric content to fabricate; count neutral
        # (2) off-source claim cited to this chunk: a sentence from an unrelated document
        other = rng.choice([x for x in ans if x["gold_doc"] != r["gold_doc"]])
        caught["wrong_claim"] += not sentence_supported(other["gold_sentence"], ch.text, idf)
    return dict(n=n, clean_sentences_kept=passed_clean / n, fabricated_number_blocked=caught["wrong_number"] / n,
                off_source_claim_blocked=caught["wrong_claim"] / n)


# ------------------------------------------------------------------------ scale bench
def bench_scale(scales=(1, 10, 40), seed=42, n_queries=150, chunk_kw=None) -> list[dict]:
    chunk_kw = chunk_kw or dict(max_words=80, pack=True)
    from .evalsets import build_eval_sets
    from .synth import generate_corpus
    out = []
    for sc in scales:
        docs, facts, users = generate_corpus(seed, sc)
        chunks, _ = ingest(docs, overlap_sentences=1, **chunk_kw)
        t0 = time.perf_counter()
        pipe = RAGPipeline(chunks, config=PipelineConfig(abstain_threshold=0.0))
        build = time.perf_counter() - t0
        urs = {u.user_id: u for u in users}
        rows = [r for r in build_eval_sets(docs, facts, users, seed=1, n_answerable=n_queries, n_forbidden=0)
                if r["kind"] == "answerable"]
        gold = GoldIndex(chunks)
        lat, hits5 = [], []
        for r in rows:
            u = urs[r["user_id"]]
            t = time.perf_counter()
            pipe.ask(u, r["question"])
            lat.append((time.perf_counter() - t) * 1e3)
            h = pipe.retriever.retrieve(r["question"], pipe.acl.for_user(u), 5)
            rel = gold.relevant(r)
            hits5.append(hit_at_k([x.idx in rel for x in h], 5))
        out.append(dict(scale=sc, documents=len(docs), chunks=len(chunks), build_s=round(build, 2),
                        p50_ms=round(float(np.percentile(lat, 50)), 2), p95_ms=round(float(np.percentile(lat, 95)), 2),
                        recall_at_5=round(float(np.mean(hits5)), 3)))
        print("scale", out[-1], flush=True)
    return out
