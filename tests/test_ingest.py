from erag.ingest import chunk_document, ingest, looks_like_injection
from erag.models import Classification as C, Document


def test_poisoned_documents_are_quarantined(world):
    assert set(world["report"].quarantined) == {"pub-poison-0", "pub-poison-1", "pub-poison-2"}
    assert not any(c.doc_id.startswith("pub-poison") for c in world["chunks"])


def test_injection_scanner_does_not_flag_normal_policy_text(world):
    clean = [d for d in world["docs"] if not d.doc_id.startswith("pub-poison")]
    assert not any(looks_like_injection(d.text) for d in clean)


def test_chunks_inherit_acl_metadata_exactly(world):
    by_id = {d.doc_id: d for d in world["docs"]}
    for c in world["chunks"]:
        d = by_id[c.doc_id]
        assert (c.classification, c.department, c.allowed_groups) == (d.classification, d.department, d.allowed_groups)


def test_sentence_overlap_and_budget():
    text = "# T\n## S\n" + " ".join(f"Sentence number {i} has exactly seven words here." for i in range(10))
    d = Document("d", "T", text, "HR", C.INTERNAL)
    cs = chunk_document(d, max_words=30, overlap_sentences=1)
    assert len(cs) > 1
    assert all(len(c.text.split()) <= 30 for c in cs)
    assert cs[0].text.split(". ")[-1] in cs[1].text  # last sentence repeated -> fact not cut at a boundary


def test_no_sentence_is_lost():
    text = "# T\n## A\nAlpha one. Alpha two.\n## B\nBeta one. Beta two."
    d = Document("d", "T", text, "HR", C.INTERNAL)
    joined = " ".join(c.text for c in chunk_document(d, 8, 0, pack=True))
    for s in ("Alpha one.", "Alpha two.", "Beta one.", "Beta two."):
        assert s in joined
