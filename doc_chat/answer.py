"""Stage 5 — GENERATE.

Two modes:
  • Retrieval-only (default): return the top passages with their sources.
    Honest, instant, zero setup — and it already answers most factual
    questions, because the answer is literally quoted from the source.
  • --llm: ask a chat model to write a grounded answer using ONLY the
    retrieved passages. Works with OpenAI or a local Ollama (same API shape).

Grounding is the point: we hand the model the evidence, not its memory. That
is what stops it inventing things — this is 'RAG' vs 'hallucination'.
"""
from __future__ import annotations

import os

from .store import Hit


SYSTEM_PROMPT = (
    "You answer strictly from the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Cite the sources you used like [1], [2]."
)


def build_context(hits: list[Hit]) -> str:
    blocks = [f"[{h.rank}] {h.source}\n{h.chunk}" for h in hits]
    return "\n\n".join(blocks)


def format_retrieval_only(query: str, hits: list[Hit]) -> str:
    lines = [f"Question: {query}", "", "Top passages by cosine similarity:", ""]
    for h in hits:
        snippet = h.chunk if len(h.chunk) <= 600 else h.chunk[:600] + " ..."
        lines.append(f"[{h.rank}] {h.source}  (score {h.score:.3f})")
        lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def answer_with_llm(query: str, hits: list[Hit]) -> str:
    """Grounded answer via any OpenAI-compatible endpoint.

    OpenAI:   set OPENAI_API_KEY (uses api.openai.com)
    Ollama:   run `ollama serve` + `ollama pull llama3.1`, then set
              OPENAI_BASE_URL=http://localhost:11434/v1 and OPENAI_API_KEY=ollama
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("LLM mode needs the openai package:  pip install openai") from exc

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    context = build_context(hits)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content or ""
