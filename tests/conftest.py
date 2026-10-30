import pytest

from erag.access import AccessMask, UserDirectory
from erag.audit import AuditLog
from erag.ingest import ingest
from erag.pipeline import PipelineConfig, RAGPipeline
from erag.synth import generate_corpus


@pytest.fixture(scope="session")
def world():
    docs, facts, users = generate_corpus(42)
    chunks, report = ingest(docs, max_words=80, overlap_sentences=1, pack=True)
    pipe = RAGPipeline(chunks, config=PipelineConfig(abstain_threshold=0.33))
    return dict(docs=docs, facts=facts, users={u.user_id: u for u in users}, chunks=chunks, report=report, pipe=pipe,
                directory=UserDirectory(users))
