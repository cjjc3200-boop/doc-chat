#!/usr/bin/env python3
"""Ask questions against your docs.

Usage:
    python chat.py                      # retrieval-only (no API key needed)
    python chat.py --llm                # grounded answer via an LLM
    python chat.py --llm --once "your question"
    python chat.py --top-k 6
"""
from __future__ import annotations

import argparse

from doc_chat.config import TOP_K
from doc_chat.pipeline import answer_query


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your docs (local RAG).")
    ap.add_argument("--llm", action="store_true", help="synthesise an answer with an LLM instead of listing passages")
    ap.add_argument("--top-k", type=int, default=TOP_K, help=f"passages to retrieve (default {TOP_K})")
    ap.add_argument("--once", metavar="QUESTION", help="ask a single question and exit")
    args = ap.parse_args()

    if args.once:
        print(answer_query(args.once, k=args.top_k, use_llm=args.llm))
        return

    print("Ask questions about your docs. Type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        print()
        print(answer_query(q, k=args.top_k, use_llm=args.llm))
        print()


if __name__ == "__main__":
    main()
