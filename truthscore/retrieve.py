"""
Evidence retrieval: TF-IDF cosine (always available) and optional FAISS + embeddings.

Pass a custom retriever to ``TruthScorer(retriever=...)`` backed by your corpus,
vector DB, or search API.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Protocol, Sequence, Tuple


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_sparse(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[w] * b[w] for w in a if w in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class TfidfIndex:
    """In-memory TF–IDF vectors for a fixed passage corpus."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.documents = list(documents)
        n = max(1, len(self.documents))
        self._df: Counter = Counter()
        self._doc_tf: List[Counter] = []
        for d in self.documents:
            tf = Counter(_tokens(d))
            self._doc_tf.append(tf)
            for w in tf:
                self._df[w] += 1
        self._idf = {w: math.log((n + 1) / (df + 1)) + 1.0 for w, df in self._df.items()}

    def _tfidf(self, tf: Counter) -> Counter:
        total = sum(tf.values()) or 1
        vec: Counter = Counter()
        for w, c in tf.items():
            if w in self._idf:
                vec[w] = (c / total) * self._idf[w]
        return vec

    def vectorize(self, text: str) -> Counter:
        return self._tfidf(Counter(_tokens(text)))

    def doc_vector(self, i: int) -> Counter:
        return self._tfidf(self._doc_tf[i])

    def similarity(self, a: str, b: str) -> float:
        return _cosine_sparse(self.vectorize(a), self.vectorize(b))

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        qv = self.vectorize(query)
        scored: List[Tuple[int, float]] = []
        for i in range(len(self.documents)):
            scored.append((i, _cosine_sparse(qv, self.doc_vector(i))))
        scored.sort(key=lambda x: -x[1])
        return scored[: max(1, k)]


class EvidenceRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...


class TfidfPassageRetriever:
    """Retrieve top-k passages from a corpus by TF–IDF cosine similarity."""

    def __init__(self, passages: Sequence[str], *, source_prefix: str = "corpus") -> None:
        if not passages:
            raise ValueError("TfidfPassageRetriever requires a non-empty corpus.")
        self._index = TfidfIndex(passages)
        self._prefix = source_prefix

    @property
    def index(self) -> TfidfIndex:
        return self._index

    def similarity(self, a: str, b: str) -> float:
        return self._index.similarity(a, b)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self._index.top_k(query, top_k)
        out: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(hits):
            out.append(
                {
                    "text": self._index.documents[idx],
                    "source": f"{self._prefix}:{idx}",
                    "relevance": float(max(0.0, min(1.0, score))),
                    "rank": rank,
                }
            )
        return out


def build_faiss_retriever(
    passages: Sequence[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    source_prefix: str = "faiss",
):
    """
    Optional FAISS + sentence-transformers retriever.

    Install: ``pip install 'truthscore-llm[retrieval]'``.
    """
    try:
        import faiss  # type: ignore
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "FAISS retriever requires faiss-cpu, sentence-transformers, numpy."
        ) from e

    model = SentenceTransformer(model_name)
    texts = list(passages)
    embs = model.encode(texts, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embs, dtype=np.float32))

    class _FaissRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            q = model.encode([query], normalize_embeddings=True)
            scores, idxs = index.search(np.asarray(q, dtype=np.float32), top_k)
            out: List[Dict[str, Any]] = []
            for rank, (idx, sc) in enumerate(zip(idxs[0], scores[0])):
                if idx < 0:
                    continue
                out.append(
                    {
                        "text": texts[int(idx)],
                        "source": f"{source_prefix}:{int(idx)}",
                        "relevance": float(max(0.0, min(1.0, (float(sc) + 1) / 2))),
                        "rank": rank,
                    }
                )
            return out

    return _FaissRetriever()


def compute_retrieval_coverage(
    claim_records: Sequence[Any],
    *,
    min_relevance: float = 0.08,
) -> float:
    """Mean top relevance across claims (0 if nothing retrieved)."""
    rels: List[float] = []
    for rec in claim_records:
        ev = getattr(rec, "evidence", None) or []
        if not ev:
            continue
        best = max((float(d.get("relevance", 0.0)) for d in ev), default=0.0)
        if best >= min_relevance:
            rels.append(best)
    if not rels:
        return 0.0
    return float(sum(rels) / len(rels))


# Backwards-compatible entry point used by older examples
def retrieve_evidence(question: str, answer: str, top_k: int = 5) -> List[Dict[str, Any]]:
    from truthscore.default_corpus import DEFAULT_PASSAGES

    r = TfidfPassageRetriever(DEFAULT_PASSAGES)
    q = f"{question}\n{answer}".strip() or question
    return r.retrieve(q, top_k)
