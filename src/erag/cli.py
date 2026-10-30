"""CLI:  python -m erag.cli {demo|ask|token|serve}"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .access import UserDirectory
from .ingest import ingest, load_documents, save_documents
from .pipeline import PipelineConfig, RAGPipeline
from .synth import generate_corpus

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CHUNKING = dict(max_words=80, overlap_sentences=1, pack=True)


def _ensure_data() -> None:
    if not (DATA / "corpus.jsonl").exists():
        DATA.mkdir(exist_ok=True)
        docs, _, users = generate_corpus(42)
        save_documents(docs, DATA / "corpus.jsonl")
        UserDirectory(users).to_json(DATA / "users.json")


def build_pipeline(threshold: float | None = None) -> tuple[RAGPipeline, UserDirectory]:
    _ensure_data()
    chunks, rep = ingest(load_documents(DATA / "corpus.jsonl"), **CHUNKING)
    if rep.quarantined:
        print(f"[ingest] quarantined {len(rep.quarantined)} document(s) for injection content: {rep.quarantined}",
              file=sys.stderr)
    cfg = PipelineConfig(abstain_threshold=threshold if threshold is not None else _default_threshold())
    from .audit import AuditLog
    audit = AuditLog(ROOT / "artifacts/audit.jsonl")
    return RAGPipeline(chunks, config=cfg, audit=audit), UserDirectory.from_json(DATA / "users.json")


def _default_threshold() -> float:
    f = ROOT / "artifacts/results.json"
    return json.loads(f.read_text())["threshold"] if f.exists() else 0.33


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="erag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("--user", required=True); a.add_argument("question")
    t = sub.add_parser("token"); t.add_argument("user")
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    sub.add_parser("demo")
    args = ap.parse_args(argv)

    if args.cmd == "token":
        from .api import issue_token
        print(issue_token(args.user, os.environ["ERAG_JWT_SECRET"]))
        return
    pipe, directory = build_pipeline()
    if args.cmd == "ask":
        user = directory.get(args.user) or sys.exit(f"unknown user {args.user}")
        ans = pipe.ask(user, args.question)
        print(ans.text)
        for c in ans.citations:
            print(f"  [{c.n}] {c.title} > {c.section}  ({c.chunk_id})")
        print(f"  confidence={ans.confidence:.2f} latency={ans.latency_ms:.1f}ms abstained={ans.abstained}")
    elif args.cmd == "demo":
        q = "How many days notice is required to terminate the Zentara Systems agreement?"
        for uid in ("erin", "ivan", "dave", "henry"):
            u = directory.get(uid)
            ans = pipe.ask(u, q)
            print(f"\n[{u.name}] clearance={u.clearance.name}\n  Q: {q}\n  A: {ans.text}")
            for c in ans.citations:
                print(f"     [{c.n}] {c.title} ({c.chunk_id})")
    elif args.cmd == "serve":
        import uvicorn
        from .api import create_app
        secret = os.environ.get("ERAG_JWT_SECRET") or sys.exit("set ERAG_JWT_SECRET (>=16 chars)")
        uvicorn.run(create_app(pipe, directory, secret), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
