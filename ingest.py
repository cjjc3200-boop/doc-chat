#!/usr/bin/env python3
"""Build the vector index from everything in docs/.

Usage:
    python ingest.py
"""
from doc_chat.pipeline import build_index

if __name__ == "__main__":
    print("Ingesting docs/…")
    build_index()
    print("Done. Now run:  python chat.py")
