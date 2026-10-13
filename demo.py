#!/usr/bin/env python3
"""Non-interactive demo — runs a few sample queries so you can see retrieval work."""
from doc_chat.pipeline import answer_query

QUESTIONS = [
    "How much does a website cost?",
    "What is the AI Lead Qualifier?",
    "Do you do social media management?",
    "What happens after I book a call?",
]

if __name__ == "__main__":
    for q in QUESTIONS:
        print("=" * 78)
        print(answer_query(q))
        print()
