"""doc_chat — a minimal, teachable RAG pipeline you can run locally.

Each module maps to one concept in the RAG vocabulary:
    chunking  -> chunk
    embedder  -> embed (vector)
    store     -> index + retrieve (cosine similarity)
    answer    -> generate (grounded answer with citations)
"""

__version__ = "0.1.0"
