"""Stage 1 — CHUNKING.

Why chunk at all? An embedding compresses text into ONE vector. A whole
1,000-page handbook squeezed into a single vector loses almost everything,
and a tiny 5-word scrap has no context. So we split docs into passage-sized
chunks (~a paragraph) that are self-contained enough to be useful on their own.

Overlap matters: if a key sentence straddles a chunk boundary, overlap keeps
it whole in at least one chunk.
"""
from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for a demo and easy to read."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring to end on a sentence.

    Roughly: walk forward by chunk_size, nudge the cut to the next sentence
    boundary, then step back by `overlap` so the next chunk re-reads the tail.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    start, n = 0, len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # Try to end on a sentence boundary near the target end.
        if end < n:
            nxt = text.find(". ", end)
            if nxt != -1 and nxt - end < 200:
                end = nxt + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)  # step back for overlap

    return chunks
