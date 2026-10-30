"""Hybrid retrieval: BM25 (sparse) + dense vectors, fused with Reciprocal Rank Fusion,
then a cheap coverage reranker. The ACL mask is applied INSIDE every ranker before top-k
selection, so unauthorised chunks can neither be returned nor consume top-k slots.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import Chunk
from .textutil import tokenize


# ----------------------------------------------------------------------------- sparse
class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}

    def fit(self, docs_tokens: list[list[str]]) -> "BM25Index":
        rows, cols, data = [], [], []
        df: Counter = Counter()
        dl = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        for i, toks in enumerate(docs_tokens):
            for t, c in Counter(toks).items():
                j = self.vocab.setdefault(t, len(self.vocab))
                rows.append(i); cols.append(j); data.append(c); df[j] += 1
        n, v = len(docs_tokens), len(self.vocab)
        tf = sp.coo_matrix((data, (rows, cols)), shape=(n, v), dtype=np.float32)
        idf = np.zeros(v, dtype=np.float32)
        for j, d in df.items():
            idf[j] = np.log(1 + (n - d + 0.5) / (d + 0.5))
        avgdl = max(float(dl.mean()), 1e-9)
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=(n, v), dtype=np.float32)
        self.idf = idf
        self.n = n
        self._unknown_idf = float(np.log(1 + (n + 0.5) / 0.5))
        return self

    def idf_of(self, term: str) -> float:
        j = self.vocab.get(term)
        return float(self.idf[j]) if j is not None else self._unknown_idf

    def scores(self, q_tokens: list[str]) -> np.ndarray:
        idx = sorted({self.vocab[t] for t in q_tokens if t in self.vocab})
        if not idx:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.W[:, idx].sum(axis=1), dtype=np.float32).ravel()


# ------------------------------------------------------------------------------ dense
class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    """Dependency-free dense embedder (TF-IDF -> truncated SVD, i.e. latent semantic analysis).

    It runs offline so the whole project is reproducible without model downloads. It is a
    stand-in for a neural encoder: swap in `SentenceTransformerEmbedder` for production.
    """

    def __init__(self, dim: int = 192, seed: int = 7):
        self.dim, self.seed = dim, seed

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     sublinear_tf=True, max_features=60000)
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed, algorithm="randomized", n_iter=4)
        self.svd.fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        Z = self.svd.transform(self.tfidf.transform(texts)).astype(np.float32)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.maximum(norms, 1e-9)


class WordVectorEmbedder:
    """Semantic embedder from pretrained static word vectors (spaCy `en_core_web_md`, GloVe-style):
    IDF-weighted mean of word vectors. Bridges vocabulary mismatch (pay ~ salary, cost ~ price) that
    TF-IDF/LSA cannot. Optional dependency: `pip install spacy` + the model wheel. Rare entity names are
    out-of-vocabulary here on purpose; BM25 covers exact names."""

    def __init__(self, model: str = "en_core_web_md"):
        import spacy  # lazy import
        self.nlp = spacy.load(model, disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
        self.vocab = self.nlp.vocab
        self.dim = self.vocab.vectors.shape[1]
        self._cache: dict[str, np.ndarray | None] = {}
        self.idf: dict[str, float] = {}
        self.default_idf = 1.0

    @staticmethod
    def _words(text: str) -> list[str]:
        import re
        from .textutil import STOP
        return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 1]

    def _vec(self, w: str):
        if w not in self._cache:
            self._cache[w] = self.vocab[w].vector if self.vocab.has_vector(w) else None
        return self._cache[w]

    def fit(self, texts: list[str]) -> "WordVectorEmbedder":
        df: Counter = Counter()
        for t in texts:
            df.update(set(self._words(t)))
        n = len(texts)
        self.idf = {w: float(np.log(1 + n / c)) for w, c in df.items()}
        self.default_idf = float(np.median(list(self.idf.values()))) if self.idf else 1.0
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in self._words(t):
                v = self._vec(w)
                if v is not None:
                    out[i] += self.idf.get(w, self.default_idf) * v
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    """Production adapter (not used in the offline benchmark). Requires `sentence-transformers`
    and network access to fetch weights, e.g. 'BAAI/bge-small-en-v1.5'."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):  # no training needed
        return self

    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32)


# --------------------------------------------------------------------------- retrieval
@dataclass
class Hit:
    idx: int
    score: float
    chunk: Chunk
    bm25_rank: int | None = None
    dense_rank: int | None = None


def _top_k(scores: np.ndarray, mask: np.ndarray | None, k: int, min_score: float = -np.inf) -> np.ndarray:
    s = scores if mask is None else np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum())) if mask is not None else min(k, len(s))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-s, k - 1)[:k]
    part = part[np.argsort(-s[part])]
    return part[s[part] > min_score]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, use_header: bool = True,
                 rrf_k: int = 60, n_candidates: int = 50, rerank_weight: float = 1.0):
        self.chunks = chunks
        self.use_header = use_header
        self.rrf_k, self.n_cand, self.rerank_weight = rrf_k, n_candidates, rerank_weight
        texts = [c.indexed_text if use_header else c.text for c in chunks]
        self.bm25 = BM25Index().fit([tokenize(t) for t in texts])
        self.embedder = embedder or LSAEmbedder()
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)
        self._terms = [frozenset(tokenize(t)) for t in texts]

    # -- ranking primitives ---------------------------------------------------------
    def bm25_rank(self, q_tokens, mask, k):
        return _top_k(self.bm25.scores(q_tokens), mask, k, min_score=0.0)

    def dense_rank(self, query, mask, k):
        qv = self.embedder.encode([query])[0]
        return _top_k(self.vectors @ qv, mask, k)

    def coverage(self, q_tokens: list[str], idx: int) -> float:
        """IDF-weighted share of the query's informative terms present in a chunk (0..1).
        Also used as the answerability signal."""
        qs = set(q_tokens)
        if not qs:
            return 0.0
        tot = sum(self.bm25.idf_of(t) for t in qs)
        got = sum(self.bm25.idf_of(t) for t in qs if t in self._terms[idx])
        return got / tot if tot > 0 else 0.0

    # -- public API -----------------------------------------------------------------
    def retrieve(self, query: str, mask: np.ndarray | None, k: int = 5, mode: str = "hybrid+rerank",
                 timings: dict | None = None) -> list[Hit]:
        t0 = time.perf_counter()
        q_tokens = tokenize(query)
        sparse = dense = np.empty(0, dtype=np.int64)
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            sparse = self.bm25_rank(q_tokens, mask, self.n_cand)
        t1 = time.perf_counter()
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            dense = self.dense_rank(query, mask, self.n_cand)
        t2 = time.perf_counter()

        if mode == "bm25":
            order = [(int(i), 0.0) for i in sparse[:k]]
        elif mode == "dense":
            order = [(int(i), 0.0) for i in dense[:k]]
        else:
            rrf: dict[int, float] = {}
            for r, i in enumerate(sparse):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.rrf_k + r + 1)
            for r, i in enumerate(dense):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.rrf_k + r + 1)
            if mode == "hybrid+rerank" and rrf:
                top = max(rrf.values())
                rrf = {i: v / top + self.rerank_weight * self.coverage(q_tokens, i) for i, v in rrf.items()}
            order = sorted(rrf.items(), key=lambda kv: -kv[1])[:k]
        t3 = time.perf_counter()
        if timings is not None:
            timings.update(bm25_ms=(t1 - t0) * 1e3, dense_ms=(t2 - t1) * 1e3, fuse_ms=(t3 - t2) * 1e3)

        sp_rank = {int(i): r for r, i in enumerate(sparse)}
        de_rank = {int(i): r for r, i in enumerate(dense)}
        return [Hit(i, s, self.chunks[i], sp_rank.get(i), de_rank.get(i)) for i, s in order]
