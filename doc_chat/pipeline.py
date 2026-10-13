"""Wire the stages together: build the index, and answer a query."""
from __future__ import annotations

from pathlib import Path

from .answer import answer_with_llm, format_retrieval_only, build_context
from .chunking import chunk_text
from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, INDEX_PATH, TOP_K
from .embedder import Embedder
from .loader import iter_docs
from .store import VectorStore


def build_index(docs_dir: Path = DOCS_DIR, index_path: Path = INDEX_PATH) -> VectorStore:
    embedder = Embedder()
    docs = iter_docs(docs_dir)
    if not docs:
        raise SystemExit(f"No documents found in {docs_dir}. Add .md/.txt (or .pdf) files there.")

    chunks: list[str] = []
    sources: list[str] = []
    for source, text in docs:
        for c in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(c)
            sources.append(source)

    print(f"  loaded {len(docs)} document(s)")
    print(f"  produced {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embeddings = embedder.embed(chunks)
    print(f"  embedded -> {embeddings.shape[0]} vectors of dim {embeddings.shape[1]}")

    store = VectorStore(embeddings, chunks, sources)
    store.save(index_path)
    print(f"  index saved to {index_path}.npy / .json")
    return store


def answer_query(query: str, index_path: Path = INDEX_PATH, k: int = TOP_K, use_llm: bool = False) -> str:
    store = VectorStore.load(index_path)
    embedder = Embedder()
    query_vec = embedder.embed([query])
    hits = store.search(query_vec, k=k)
    if use_llm:
        return answer_with_llm(query, hits)
    return format_retrieval_only(query, hits)
