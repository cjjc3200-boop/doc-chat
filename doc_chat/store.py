"""Stage 3 & 4 — INDEX and RETRIEVE.

Because every vector is unit-length, cosine similarity between a query and a
stored chunk is just their dot product. So retrieval is ONE matrix multiply:

        similarities = embeddings @ query      # shape (N,)
        best = argsort(-similarities)[:k]

No vector database required to understand the idea — this IS the idea. Real
vector DBs (FAISS, Chroma, pgvector) exist because this brute-force scan is
O(N) and they add speed, filtering, and persistence at scale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hit:
    rank: int
    score: float      # cosine similarity, -1..1 (higher = more similar)
    chunk: str
    source: str       # which file it came from


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[str], sources: list[str]) -> None:
        assert len(chunks) == len(sources) == embeddings.shape[0], "metadata/vector count mismatch"
        self.embeddings = embeddings          # (N, D) unit vectors
        self.chunks = chunks
        self.sources = sources

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.embeddings)
        path.with_suffix(".json").write_text(
            json.dumps({"chunks": self.chunks, "sources": self.sources}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        path = Path(path)
        embeddings = np.load(path.with_suffix(".npy"))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(embeddings, meta["chunks"], meta["sources"])

    # ---------- retrieval ----------
    def search(self, query_vec: np.ndarray, k: int = 4) -> list[Hit]:
        """query_vec: (1, D) or (D,) unit vector. Returns top-k Hits."""
        q = query_vec.reshape(-1)                 # (D,)
        sims = self.embeddings @ q                # (N,) cosine similarity
        k = min(k, len(sims))
        top = np.argsort(-sims)[:k]
        return [
            Hit(rank=i + 1, score=float(sims[j]), chunk=self.chunks[j], source=self.sources[j])
            for i, j in enumerate(top)
        ]
