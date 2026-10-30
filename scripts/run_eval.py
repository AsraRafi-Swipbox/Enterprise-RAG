#!/usr/bin/env python
"""Reproduce every number in the README:  PYTHONPATH=src python scripts/run_eval.py
Design rule: every choice (chunking, abstention threshold) is made on the DEV split; all headline
numbers are reported on the held-out TEST split."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from erag.access import AccessMask, UserDirectory  # noqa: E402
from erag.evalsets import build_eval_sets  # noqa: E402
from erag.evaluation import (GoldIndex, bench_scale, calibrate, eval_acl_strategies, eval_retrieval,  # noqa: E402
                             eval_verifier, run_e2e, score_at)
from erag.generation import ExtractiveGenerator  # noqa: E402
from erag.ingest import ingest, save_documents  # noqa: E402
from erag.pipeline import PipelineConfig, RAGPipeline  # noqa: E402
from erag.retrieval import HybridRetriever  # noqa: E402
from erag.synth import generate_corpus  # noqa: E402

SEED = 42
for d in ("data", "artifacts", "docs"):
    (ROOT / d).mkdir(exist_ok=True)

try:
    from erag.retrieval import WordVectorEmbedder
    WordVectorEmbedder()
    HAVE_WV = True
except Exception:  # spaCy / model not installed: optional rows are skipped
    HAVE_WV = False

# ------------------------------------------------------------------ data
docs, facts, users = generate_corpus(SEED)
urs = {u.user_id: u for u in users}
rows = build_eval_sets(docs, facts, users)
save_documents(docs, ROOT / "data/corpus.jsonl")
UserDirectory(users).to_json(ROOT / "data/users.json")
(ROOT / "data/eval_queries.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
answerable = [r for r in rows if r["kind"] == "answerable"]
dev_a = [r for r in answerable if r["split"] == "dev"]
test_a = [r for r in answerable if r["split"] == "test"]
kinds = sorted({r["kind"] for r in rows})
print(f"{len(docs)} docs, {len(facts)} facts, {len(rows)} queries; dev/test answerable {len(dev_a)}/{len(test_a)}", flush=True)
R: dict = {"seed": SEED, "corpus": {"documents": len(docs), "facts": len(facts), "queries": len(rows),
                                     "by_kind": {k: sum(r["kind"] == k for r in rows) for k in kinds},
                                     "avg_words_per_doc": float(np.mean([len(d.text.split()) for d in docs]))},
           "have_wordvec": HAVE_WV}


def e2e_variant(chunks, retr, gen=None, entity_gate=True):
    pipe = RAGPipeline(chunks, retriever=retr, generator=gen, config=PipelineConfig(abstain_threshold=0.0,
                                                                                    entity_gate=entity_gate))
    recs = run_e2e(pipe, GoldIndex(chunks), urs, rows)
    dev = [r for r in recs if r["split"] == "dev"]
    test = [r for r in recs if r["split"] == "test"]
    thr = calibrate(dev)
    return pipe, recs, thr, score_at(dev, thr), score_at(test, thr)


# ------------------------------------------------- 1. chunking / context ablation (selected on DEV only)
CONFIGS = [("section(<=120w)", dict(max_words=120, pack=False)), ("pack 40w", dict(max_words=40, pack=True)),
           ("pack 80w", dict(max_words=80, pack=True)), ("pack 160w", dict(max_words=160, pack=True))]
abl = []
for name, kw in CONFIGS:
    for header in (True, False):
        ch, _ = ingest(docs, overlap_sentences=1, **kw)
        retr = HybridRetriever(ch, use_header=header)
        m = eval_retrieval(retr, AccessMask(ch), GoldIndex(ch), urs, dev_a, "hybrid+rerank")
        pipe = RAGPipeline(ch, retriever=retr, config=PipelineConfig(abstain_threshold=0.0))
        recs = [r for r in run_e2e(pipe, GoldIndex(ch), urs, [r for r in rows if r["split"] == "dev"])]
        gc = score_at(recs, 0.0)["grounded_correct"]
        abl.append(dict(config=name, header=header, chunks=len(ch), avg_words=float(np.mean([len(c.text.split()) for c in ch])),
                        mrr=m["mrr"], recall1=m["recall@1"], recall5=m["recall@5"], doc_recall1=m["doc_recall@1"],
                        grounded_correct=gc))
        print("ablation", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in abl[-1].items()}, flush=True)
R["ablation_dev"] = abl
best = max((a for a in abl if a["header"]), key=lambda a: (a["grounded_correct"], a["mrr"]))
best_kw = dict(CONFIGS)[best["config"]]
R["chosen_chunking"] = best["config"]
print("chosen chunking:", best["config"], flush=True)

chunks, rep = ingest(docs, overlap_sentences=1, **best_kw)
R["ingest"] = dict(documents=rep.documents, chunks=rep.chunks, quarantined=rep.quarantined,
                   avg_words_per_chunk=float(np.mean([len(c.text.split()) for c in chunks])))
retr = HybridRetriever(chunks)
acl, gold = AccessMask(chunks), GoldIndex(chunks)

# ------------------------------------------------- 2. retrieval modes on held-out TEST
R["retrieval_test"] = {}
for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):
    R["retrieval_test"][mode] = m = eval_retrieval(retr, acl, gold, urs, test_a, mode)
    print(f"retrieval {mode:14s} R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} MRR {m['mrr']:.3f} "
          f"docR@1 {m['doc_recall@1']:.3f}", flush=True)

# ------------------------------------------------- 3. access-control strategies (all queries)
R["acl"] = eval_acl_strategies(retr, acl, gold, urs, rows)
print("acl", json.dumps(R["acl"]), flush=True)

# ------------------------------------------------- 4. end-to-end, threshold calibrated on DEV
pipe, recs, thr, s_dev, s_test = e2e_variant(chunks, retr)
test = [r for r in recs if r["split"] == "test"]
R["threshold"], R["e2e_dev"], R["e2e_test"] = thr, s_dev, s_test
R["e2e_test_no_abstention"] = score_at(test, 0.0)
R["threshold_sweep_test"] = [score_at(test, t) for t in np.round(np.arange(0.0, 1.001, 0.05), 2)]
R["by_style_test"] = {}
for st, nm in enumerate(["direct", "paraphrase", "keyword"]):
    g = [r for r in test if r["kind"] == "answerable" and r["style"] == st]
    R["by_style_test"][nm] = dict(n=len(g), grounded_correct=float(np.mean([r["value_ok"] and r["cite_ok"] and not r["abstained0"] and r["conf"] >= thr for r in g])))
R["invariants"] = dict(queries=len(recs), unauthorised_citations=sum(r["unauth_cited"] for r in recs),
                       adversarial_queries=sum(r["kind"] == "adversarial" for r in recs),
                       adversarial_unauthorised_citations=sum(r["unauth_cited"] for r in recs if r["kind"] == "adversarial"))
lat = np.array([r["latency_ms"] for r in recs])
R["latency"] = dict(p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
                    p99_ms=float(np.percentile(lat, 99)),
                    mean_stage_ms={k: float(np.mean([r["trace"].get(k, 0.0) for r in recs])) for k in
                                   ("mask_ms", "bm25_ms", "dense_ms", "fuse_ms", "generate_ms")})
print("threshold", thr, "test", json.dumps({k: round(v, 3) for k, v in s_test.items()}), flush=True)

# ------------------------------------------------- 5. component ablation (each variant re-calibrated on DEV)
comp = {"Full system": dict(dev=s_dev, test=s_test)}
def add(name, chunks_, retr_, gen=None, gate=True):
    _, _, t, d, te = e2e_variant(chunks_, retr_, gen, gate)
    comp[name] = dict(dev=d, test=te, threshold=t)
    print("component", name, {k: round(v, 3) for k, v in te.items()}, flush=True)
add("- entity gate (coverage only)", chunks, retr, gate=False)
add("- embedding in sentence scorer (lexical)", chunks, retr, gen=ExtractiveGenerator(), gate=True)
add("- contextual chunk header", chunks, HybridRetriever(chunks, use_header=False))
if HAVE_WV:
    from erag.retrieval import WordVectorEmbedder
    wv = WordVectorEmbedder().fit([c.indexed_text for c in chunks])
    add("+ word-vector sentence scorer (spaCy md)", chunks, retr, gen=ExtractiveGenerator(embedder=wv, alpha=0.7))
    add("dense channel = word vectors (instead of LSA)", chunks, HybridRetriever(chunks, embedder=WordVectorEmbedder()))
comp["No abstention gate"] = dict(dev=score_at([r for r in recs if r["split"] == "dev"], 0.0), test=R["e2e_test_no_abstention"])
R["component_ablation"] = comp

# ------------------------------------------------- 6. citation verifier fuzzing
R["verifier"] = eval_verifier(pipe, gold, rows)
print("verifier", R["verifier"], flush=True)

# ------------------------------------------------- 7. scale benchmark
R["scale"] = bench_scale((1, 10, 40), chunk_kw=best_kw)

(ROOT / "artifacts/results.json").write_text(json.dumps(R, indent=2, default=float))
print("wrote artifacts/results.json", flush=True)
