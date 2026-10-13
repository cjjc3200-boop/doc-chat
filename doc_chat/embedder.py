"""Stage 2 — EMBEDDING.

An embedding turns a piece of text into a fixed-length list of numbers (a
vector) whose DIRECTION encodes meaning. Texts that mean similar things end up
pointing in similar directions. That is the whole trick behind semantic search:
we can compare meaning with geometry.

We use fastembed (ONNX, ~130MB) so there is NO PyTorch dependency — a much
lighter install than sentence-transformers. We L2-normalise every vector so
that cosine similarity reduces to a plain dot product (see store.py).
"""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from .config import EMBED_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model_name = model_name
        # Downloads + caches the model on first run.
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 array of L2-normalised vectors."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return _l2_normalize(vectors)

    @property
    def dim(self) -> int:
        return 384  # bge-small-en-v1.5 output dimension


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
