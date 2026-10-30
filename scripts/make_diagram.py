#!/usr/bin/env python
"""Generates docs/architecture.svg (system overview). Run: python scripts/make_diagram.py"""
from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1180, 760
out = []
add = out.append

PAL = {  # fill, stroke
    "ing": ("#E8F1FB", "#2F6DB5"), "srv": ("#E7F6F2", "#1E8E75"), "sec": ("#FDECEA", "#C0392B"),
    "ev": ("#F1EAF8", "#7D4FB0"), "store": ("#FFF6DD", "#B7860B"), "ext": ("#F3F3F3", "#777"),
}


def box(x, y, w, h, title, sub="", kind="srv", bold=True):
    f, s = PAL[kind]
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
    ty = y + (h / 2 + 4 if not sub else h / 2 - 4)
    add(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="13" font-weight="{600 if bold else 400}" fill="#1a1a1a">{escape(title)}</text>')
    if sub:
        for i, line in enumerate(sub.split("\n")):
            add(f'<text x="{x + w / 2}" y="{y + h / 2 + 11 + 12 * i}" text-anchor="middle" font-size="10.5" fill="#444">{escape(line)}</text>')


def arrow(x1, y1, x2, y2, label="", color="#444", dash=False, lx=None, ly=None):
    d = ' stroke-dasharray="5 4"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{d} marker-end="url(#a)"/>')
    if label:
        add(f'<text x="{lx if lx is not None else (x1 + x2) / 2}" y="{ly if ly is not None else (y1 + y2) / 2 - 5}" text-anchor="middle" font-size="10" fill="{color}" font-style="italic">{escape(label)}</text>')


def lane(x, y, w, h, label, kind):
    f, s = PAL[kind]
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{f}" fill-opacity="0.35" stroke="{s}" stroke-width="1" stroke-dasharray="6 4"/>')
    add(f'<text x="{x + 14}" y="{y + 20}" font-size="12" font-weight="700" fill="{s}">{escape(label)}</text>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#fff"/>')
add(f'<text x="{W / 2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="#111">Enterprise RAG: architecture</text>')
add(f'<text x="{W / 2}" y="50" text-anchor="middle" font-size="12" fill="#555">Authorisation is enforced inside retrieval, before top-k, and outside the language model</text>')

# ---------------- ingestion plane
lane(20, 70, 1140, 150, "INGESTION PLANE (offline / batch)", "ing")
box(40, 105, 140, 90, "Document sources", "JSONL today\nConfluence / SharePoint /\nDrive adapters (roadmap)", "ext")
box(215, 105, 150, 90, "Injection scan", "tripwire for planted\n'ignore previous...' text\n-> quarantine + report", "sec")
box(400, 105, 170, 90, "Section-aware chunker", "sentence overlap, word budget\ncopies classification, dept,\ngroups onto every chunk", "ing")
box(605, 105, 150, 90, "Contextual header", "title > section prepended\nto indexed text", "ing")
box(790, 105, 170, 40, "BM25 index (sparse CSC)", "", "store")
box(790, 152, 170, 43, "Dense vectors (embedder)", "LSA default / neural adapter", "store")
box(985, 105, 155, 90, "ACL signature table", "one (class, dept, groups)\nsignature per chunk", "sec")
arrow(180, 150, 215, 150); arrow(365, 150, 400, 150); arrow(570, 150, 605, 150)
arrow(755, 135, 790, 125); arrow(755, 165, 790, 172); arrow(960, 150, 985, 150)

# ---------------- serving plane
lane(20, 240, 1140, 360, "SERVING PLANE (online, per request)", "srv")
box(40, 285, 130, 70, "Client / UI", "question + JWT", "ext")
box(205, 285, 170, 70, "FastAPI gateway", "JWT verify (HS256, exp)\nsubject only, no claims trusted", "srv")
box(410, 285, 170, 70, "User directory", "clearance, departments,\ngroups (source of truth)", "sec")
box(615, 285, 170, 70, "ACL mask", "boolean mask over all chunks\ncached per user signature", "sec")
arrow(170, 320, 205, 320, "ask"); arrow(375, 320, 410, 320, "sub"); arrow(580, 320, 615, 320)

box(615, 400, 170, 80, "Hybrid retriever", "BM25 + dense, mask applied\ninside each ranker, RRF fuse,\ncoverage rerank", "srv")
arrow(700, 355, 700, 400, "mask", "#C0392B", lx=724, ly=382)
add('<line x1="985" y1="195" x2="985" y2="250" stroke="#C0392B" stroke-width="1.4" stroke-dasharray="4 3"/>')
add('<line x1="985" y1="250" x2="760" y2="250" stroke="#C0392B" stroke-width="1.4" stroke-dasharray="4 3"/>')
arrow(760, 250, 760, 285, "", "#C0392B", dash=True)
add('<text x="873" y="243" font-size="10" fill="#C0392B" font-style="italic">ACL signatures</text>')

box(410, 400, 170, 80, "Answerability gate", "IDF coverage x 0.5 per\nmissing named entity;\nbelow threshold -> abstain", "srv")
box(205, 400, 170, 80, "Generator", "extractive (offline) or LLM\nsources fenced as data,\n[n] marker per sentence", "srv")
box(40, 400, 140, 80, "Citation verifier", "numbers must appear in\ncited chunk + token support\nelse sentence dropped", "sec")
arrow(615, 440, 580, 440); arrow(410, 440, 375, 440); arrow(205, 440, 180, 440)
arrow(110, 400, 110, 355, "answer + citations", "#1E8E75", lx=165, ly=382)

box(40, 520, 170, 55, "Audit log (JSONL)", "user, query hash, cited docs,\nvisible fraction, latency", "store")
box(235, 520, 200, 55, "Chunk drill-down", "GET /v1/chunks - ACL-checked,\ndenied == 404", "srv")
arrow(110, 480, 110, 520, "every request", "#B7860B", dash=True, lx=165, ly=506)
arrow(500, 355, 500, 400, "", "#444")

box(830, 285, 310, 295, "", "", "ext")
add('<text x="985" y="308" text-anchor="middle" font-size="12" font-weight="700" fill="#333">Security invariants</text>')
inv = ["1. Identity from JWT subject only;", "    permissions looked up server-side",
       "2. Deny by default: clearance ceiling AND", "    need-to-know (dept or group)",
       "3. Mask applied BEFORE top-k: no leak,", "    no starved result slots",
       "4. assert all(mask[hit]) after retrieval", "5. Retrieved text is data, never instructions",
       "6. Unsupported / fabricated sentences", "    are removed before the user sees them",
       "7. Unauthorised == non-existent (404,", "    abstain) - no existence oracle",
       "8. Raw queries not logged (hash only)"]
for i, t in enumerate(inv):
    add(f'<text x="845" y="{333 + i * 18}" font-size="11" fill="#222" font-family="Menlo, Consolas, monospace">{escape(t).replace("    ", "&#160;&#160;&#160;&#160;")}</text>')

# ---------------- evaluation plane
lane(20, 620, 1140, 120, "EVALUATION HARNESS (reproducible: scripts/run_eval.py)", "ev")
box(40, 650, 190, 70, "Synthetic corpus", "181 docs, 810 gold facts, 4 classes,\n6 depts, hard-negative siblings", "ev")
box(260, 650, 190, 70, "881 labelled queries", "answerable, forbidden, out-of-domain,\nnear-domain, adversarial, broad", "ev")
box(480, 650, 190, 70, "dev / test split", "choices on dev,\nnumbers reported on test", "ev")
box(700, 650, 220, 70, "Metrics", "Recall@k, MRR, nDCG, exposure rate,\nabstention, grounded-correct, latency", "ev")
box(950, 650, 190, 70, "results.json -> RESULTS.md", "tables, charts, README block", "ev")
arrow(230, 685, 260, 685, "", "#7D4FB0"); arrow(450, 685, 480, 685, "", "#7D4FB0"); arrow(670, 685, 700, 685, "", "#7D4FB0"); arrow(920, 685, 950, 685, "", "#7D4FB0")
add('</svg>')
Path(__file__).resolve().parents[1].joinpath("docs/architecture.svg").write_text("\n".join(out))
print("ok")
