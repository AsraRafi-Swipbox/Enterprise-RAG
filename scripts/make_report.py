#!/usr/bin/env python
"""artifacts/results.json -> docs/results.png, docs/RESULTS.md, and the headline block inside README.md."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "artifacts/results.json").read_text())
P = lambda x: f"{100 * x:.1f}%"
rt, acl, e2e, thr = R["retrieval_test"], R["acl"], R["e2e_test"], R["threshold"]
hy = rt["hybrid+rerank"]
n_test_ans = e2e["n_answerable"]

# ------------------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))
C = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

a = ax[0, 0]
modes = list(rt)
x = np.arange(len(modes)); w = 0.27
for i, (m, lab) in enumerate([("recall@1", "Recall@1"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10")]):
    v = [rt[k][m] for k in modes]
    a.bar(x + (i - 1) * w, v, w, label=lab, color=C[i])
    for xi, vi in zip(x, v):
        a.text(xi + (i - 1) * w, vi + 0.004, f"{vi:.3f}", ha="center", fontsize=7, rotation=90)
a.set_xticks(x, modes); a.set_ylim(0.7, 1.09); a.set_yticks([0.7, 0.8, 0.9, 1.0]); a.set_title("A. Retrieval quality by mode (test, n=%d)" % hy["n"]); a.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)

a = ax[0, 1]
s = list(acl); lab = ["No ACL", "Post-filter", "Pre-filter\n(this system)"]
a.bar(np.arange(3) - 0.2, [acl[k]["unauthorised_exposure_rate"] * 100 for k in s], 0.4, color="#E45756", label="Queries exposing unauthorised chunks (%)")
a2 = a.twinx(); a2.spines["right"].set_visible(True)
a2.bar(np.arange(3) + 0.2, [acl[k]["broad_relevant_results_at_k"] for k in s], 0.4, color="#54A24B", label="Relevant visible results in top-5 (broad queries)")
a.set_xticks(range(3), lab); a.set_ylabel("exposure rate (%)"); a.set_ylim(0, 62); a2.set_ylabel("relevant results @5"); a2.set_ylim(0, 6.6)
for i, k in enumerate(s):
    a.text(i - 0.2, acl[k]["unauthorised_exposure_rate"] * 100 + 1, f"{acl[k]['exposed_queries']}/{acl[k]['total_queries']}", ha="center", fontsize=8)
    a2.text(i + 0.2, acl[k]["broad_relevant_results_at_k"] + 0.1, f"{acl[k]['broad_relevant_results_at_k']:.1f}", ha="center", fontsize=8)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center", frameon=False)
a.set_title("B. Access-control strategies (881 queries)")

a = ax[0, 2]
sw = R["threshold_sweep_test"]; t = [r["threshold"] for r in sw]
a.plot(t, [r["grounded_correct"] for r in sw], label="grounded-correct (answerable)", color=C[0], lw=2)
a.plot(t, [r["abstain_negatives"] for r in sw], label="abstains on forbidden/unanswerable", color=C[2], lw=2)
a.plot(t, [r["false_refusal"] for r in sw], label="false refusal (answerable)", color=C[1], lw=2, ls="--")
a.axvline(thr, color="k", ls=":", lw=1); a.text(thr + 0.01, 0.05, f"calibrated on dev: {thr:.2f}", fontsize=8)
a.set_xlabel("abstention threshold"); a.set_ylim(0, 1.02); a.legend(fontsize=8, loc="center left"); a.set_title("C. Answerability gate trade-off (test)")

a = ax[1, 0]
st = R["by_style_test"]; names = list(st)
v = [st[k]["grounded_correct"] for k in names]
bars = a.bar(names, v, color=[C[0], "#E45756", C[2]])
for b, k, vi in zip(bars, names, v):
    a.text(b.get_x() + b.get_width() / 2, vi + 0.01, f"{vi:.0%}\n(n={st[k]['n']})", ha="center", fontsize=8)
a.set_ylim(0, 1.08); a.set_title("D. Grounded-correct answers by question style\n(offline extractive generator)")

a = ax[1, 1]
sc = R["scale"]; ch = [r["chunks"] for r in sc]
a.plot(ch, [r["p50_ms"] for r in sc], "o-", label="p50 latency (ms)", color=C[0])
a.plot(ch, [r["p95_ms"] for r in sc], "s--", label="p95 latency (ms)", color=C[1])
a.set_xscale("log"); a.set_xlabel("indexed chunks"); a.set_ylabel("end-to-end ms (excl. LLM)"); a.set_ylim(0, max(r["p95_ms"] for r in sc) * 1.6)
a3 = a.twinx(); a3.spines["right"].set_visible(True)
a3.plot(ch, [r["recall_at_5"] for r in sc], "^-", color=C[2], label="Recall@5"); a3.set_ylim(0.9, 1.01); a3.set_ylabel("Recall@5")
for r in sc:
    a3.text(r["chunks"], r["recall_at_5"] + 0.003, f"{r['recall_at_5']:.3f}", ha="center", fontsize=8, color=C[2])
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels(); a.set_xticks(ch, [f"{c:,}" for c in ch]); a.minorticks_off()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
a.set_title("E. Scaling to 6.6k chunks (1 CPU core)")

a = ax[1, 2]
comp = R["component_ablation"]; names = list(comp)[::-1]
v = [comp[k]["test"]["balanced"] for k in names]
a.barh([n.replace(" (", "\n(") for n in names], v, color=["#9D9D9D" if n != "Full system" else C[0] for n in names])
for i, vi in enumerate(v):
    a.text(vi + 0.005, i, f"{vi:.3f}", va="center", fontsize=8)
a.set_xlim(0.3, 0.95); a.set_title("F. Component ablation: balanced score (test)")
plt.tight_layout(); (ROOT / "docs").mkdir(exist_ok=True); plt.savefig(ROOT / "docs/results.png", dpi=130)

# ------------------------------------------------------------------------------- tables
def row(*c): return "| " + " | ".join(str(x) for x in c) + " |"
inv, ver, lat = R["invariants"], R["verifier"], R["latency"]
bm = rt["bm25"]; hyb = rt["hybrid+rerank"]

headline = "\n".join([
    row("Area", "Metric", "Result"), row("---", "---", "---"),
    row("Retrieval", f"Recall@1 / Recall@5 / MRR (hybrid, test n={hy['n']})",
        f"**{P(hy['recall@1'])}** / **{P(hy['recall@5'])}** / **{hy['mrr']:.3f}** (MRR 95% CI {hy['mrr_ci'][0]:.3f}-{hy['mrr_ci'][1]:.3f})"),
    row("Access control", "Queries exposing an unauthorised chunk",
        f"**{acl['pre_filter']['exposed_queries']} / {acl['pre_filter']['total_queries']}** (no-ACL baseline: {acl['no_acl']['exposed_queries']} / {acl['no_acl']['total_queries']}, {P(acl['no_acl']['unauthorised_exposure_rate'])})"),
    row("Access control", "Relevant visible results in top-5 on under-specified queries",
        f"**{acl['pre_filter']['broad_relevant_results_at_k']:.1f}** pre-filter vs {acl['post_filter']['broad_relevant_results_at_k']:.1f} post-filter"),
    row("Access control", "Adversarial / prompt-injection queries abstained",
        f"**{P(e2e['abstain_adversarial'])}** ({inv['adversarial_queries']} queries; {inv['adversarial_unauthorised_citations']} unauthorised citations)"),
    row("Answerability", "Abstains on forbidden / non-existent-entity / out-of-domain",
        f"**{P(e2e['abstain_forbidden'])}** / **{P(e2e['abstain_unanswerable_near'])}** / **{P(e2e['abstain_unanswerable_generic'])}**"),
    row("Answers", f"Grounded-correct answers (offline extractive baseline, test n={n_test_ans})",
        f"**{P(e2e['grounded_correct'])}** (false refusals {P(e2e['false_refusal'])})"),
    row("Citations", "Cited chunk contains the gold fact (of answered)", f"**{P(e2e['citation_precision'])}**"),
    row("Citation verifier", "Fabricated numbers / off-source claims blocked; clean sentences kept",
        f"**{P(ver['fabricated_number_blocked'])}** / **{P(ver['off_source_claim_blocked'])}**; {P(ver['clean_sentences_kept'])} kept"),
    row("Ingestion", "Planted prompt-injection documents quarantined", f"**{len(R['ingest']['quarantined'])} / 3**"),
    row("Latency", "End-to-end p50 / p95 (excl. LLM call), 345 chunks",
        f"**{lat['p50_ms']:.1f} ms** / **{lat['p95_ms']:.1f} ms**; p95 {R['scale'][-1]['p95_ms']:.1f} ms at {R['scale'][-1]['chunks']:,} chunks"),
])

full = []
full += ["# Evaluation results (auto-generated by `scripts/make_report.py`)\n",
         f"Seed {R['seed']}; corpus: {R['corpus']['documents']} documents / {R['corpus']['facts']} ground-truth facts / "
         f"{R['corpus']['queries']} queries ({', '.join(f'{k}: {v}' for k, v in R['corpus']['by_kind'].items())}). "
         f"Chunking and the abstention threshold ({thr:.2f}) were selected on the **dev** split; every number below is **test** unless marked dev.\n",
         "## Headline\n", headline, "\n![results](results.png)\n",
         "## Retrieval modes (test, answerable queries)\n",
         row("Mode", "Recall@1", "Recall@5", "MRR", "nDCG@10", "Doc Recall@1", "R@5 direct / paraphrase / keyword"), row(*["---"] * 7)]
for k, m in rt.items():
    bs = m["recall@5_by_style"]
    full.append(row(k, f"{m['recall@1']:.3f}", f"{m['recall@5']:.3f}", f"{m['mrr']:.3f}", f"{m['ndcg@10']:.3f}",
                    f"{m['doc_recall@1']:.3f}", f"{bs['direct']:.2f} / {bs['paraphrase']:.2f} / {bs['keyword']:.2f}"))
full += [f"\nBootstrap 95% CI (hybrid+rerank): Recall@5 {hy['recall@5_ci'][0]:.3f}-{hy['recall@5_ci'][1]:.3f}, "
         f"MRR {hy['mrr_ci'][0]:.3f}-{hy['mrr_ci'][1]:.3f}. BM25 vs hybrid MRR CIs overlap ({bm['mrr_ci'][0]:.3f}-{bm['mrr_ci'][1]:.3f} vs "
         f"{hy['mrr_ci'][0]:.3f}-{hy['mrr_ci'][1]:.3f}): **no statistically significant hybrid gain on this corpus.**\n",
         "## Access-control strategies (all queries)\n",
         row("Strategy", "Queries exposing unauthorised chunks", "Gold doc of forbidden query exposed", "Answerable Recall@5", "Relevant visible results @5 (broad)", "Broad queries under-filled"), row(*["---"] * 6)]
for k, m in acl.items():
    full.append(row(k, f"{m['exposed_queries']}/{m['total_queries']} ({P(m['unauthorised_exposure_rate'])})", m["forbidden_gold_exposure"],
                    f"{m['answerable_recall_at_k']:.3f}", f"{m['broad_relevant_results_at_k']:.2f}", P(m["broad_underfilled_rate"])))
full += ["\n## End-to-end by population (test)\n", row("Population", "Result"), row("---", "---")]
for k in ("grounded_correct", "false_refusal", "citation_precision", "abstain_forbidden", "abstain_unanswerable_generic",
          "abstain_unanswerable_near", "abstain_adversarial"):
    full.append(row(k, P(e2e[k])))
full += ["\nBy question style (grounded-correct): " + ", ".join(f"{k} {P(v['grounded_correct'])} (n={v['n']})" for k, v in R["by_style_test"].items()) + "\n",
         "## Chunking ablation (dev split; chosen by grounded-correct, then MRR)\n",
         row("Chunking", "Context header", "Chunks", "Avg words", "Recall@1", "MRR", "Grounded-correct"), row(*["---"] * 7)]
for a in R["ablation_dev"]:
    full.append(row(a["config"], "yes" if a["header"] else "no", a["chunks"], f"{a['avg_words']:.0f}", f"{a['recall1']:.3f}", f"{a['mrr']:.3f}", f"{a['grounded_correct']:.3f}"))
full += [f"\nChosen: **{R['chosen_chunking']}**. Note: synthetic documents average {R['corpus']['avg_words_per_doc']:.0f} words, so 'pack 160w' is a whole-document chunk and wins retrieval metrics trivially; it loses on end-to-end accuracy because answers must be picked from a larger span.\n",
         "## Component ablation (each variant re-calibrated on dev, scored on test)\n",
         row("Variant", "Grounded-correct", "False refusal", "Abstain: forbidden", "Abstain: near-domain", "Balanced"), row(*["---"] * 6)]
for k, v in R["component_ablation"].items():
    t_ = v["test"]
    full.append(row(k, P(t_["grounded_correct"]), P(t_["false_refusal"]), P(t_["abstain_forbidden"]), P(t_["abstain_unanswerable_near"]), f"{t_['balanced']:.3f}"))
full += ["\nWith n=240 the standard error on a proportion near 0.75 is about 2.8 pp, so differences under ~5 pp between rows are not significant.\n",
         "## Scaling\n", row("Documents", "Chunks", "Index build (s)", "p50 ms", "p95 ms", "Recall@5"), row(*["---"] * 6)]
for r in R["scale"]:
    full.append(row(r["documents"], f"{r['chunks']:,}", r["build_s"], r["p50_ms"], r["p95_ms"], f"{r['recall_at_5']:.3f}"))
full += ["\nMean per-stage latency (ms, 345 chunks): " + ", ".join(f"{k.replace('_ms', '')} {v:.2f}" for k, v in lat["mean_stage_ms"].items()) + "\n"]
(ROOT / "docs/RESULTS.md").write_text("\n".join(full))

readme = ROOT / "README.md"
if readme.exists():
    s = readme.read_text()
    s = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", "<!-- RESULTS:START -->\n" + headline + "\n<!-- RESULTS:END -->", s, flags=re.S)
    readme.write_text(s)
print("report written")
