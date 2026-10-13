# doc_chat — a local "chat with your docs" RAG tool

A small, readable Retrieval-Augmented Generation (RAG) pipeline that runs
**entirely on your machine**. No API key required to start. Built to be
*learned from* — every stage is a separate file you can read top to bottom.

It ships with a TampifyAI knowledge base, so it doubles as a sales demo:
type a question, watch it find the right passage from your own docs.

---

## The vocabulary, mapped to the code

RAG is just five stages in a loop. Each one is a concept you should be able to
define out loud:

| # | Concept | What it means | Where in the code |
|---|---------|---------------|-------------------|
| 1 | **Chunking** | Splitting documents into passage-sized pieces so each one carries a single idea | `doc_chat/chunking.py` |
| 2 | **Embedding** | Turning text into a fixed-length vector whose *direction* encodes meaning | `doc_chat/embedder.py` |
| 3 | **Index** | Storing all chunk vectors so you can search them | `doc_chat/store.py` |
| 4 | **Retrieval** | Embed the question, find the nearest chunk vectors by **cosine similarity** | `doc_chat/store.py` (`search`) |
| 5 | **Generation** | Answer using *only* the retrieved passages (grounding) | `doc_chat/answer.py` |

Key terms you'll meet here:

- **Vector** — an ordered list of numbers. `bge-small-en-v1.5` gives each chunk **384** of them.
- **Dimension** — how many numbers per vector (384 here). Bigger models use more.
- **Cosine similarity** — how aligned two vectors are, from -1 to 1. Because we
  L2-normalise every vector, it becomes a plain dot product: `embeddings @ query`.
- **Top-k** — how many nearest chunks we feed the answer step (`TOP_K`).
- **Grounding** — forcing the answer to come from retrieved evidence, not the
  model's memory. This is what reduces hallucination.
- **Context window** — the limit on how much text an LLM can read at once; the
  reason we retrieve only a few chunks instead of the whole corpus.

---

## Setup (one time)

You need **Python 3.10+**.

```bash
# create an isolated environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies (fastembed is ONNX-based, so no PyTorch download)
pip install -r requirements.txt
```

First run of `ingest.py` downloads the embedding model (~130 MB) once and
caches it. After that, everything is offline and instant.

## Run it

```bash
# 1. build the index from everything in docs/
python ingest.py

# 2. ask questions (retrieval-only — no API key needed)
python chat.py

# or run a scripted demo to see retrieval in action
python demo.py
```

Example:

```
you> how much is a website

[1] services.md  (score 0.712)
    ## Websites We build fast, mobile-friendly websites for local businesses.
    The starting price is $500 ...
```

### Optional: real generated answers (`--llm`)

Retrieval-only mode lists passages; `--llm` mode hands those passages to a chat
model and asks it to write a grounded answer.

```bash
pip install openai

# OpenAI
export OPENAI_API_KEY=sk-...
python chat.py --llm

# or fully local with Ollama (no data leaves your machine)
ollama pull llama3.1
ollama serve
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export LLM_MODEL=llama3.1
python chat.py --llm
```

## Add your own documents

Drop `.md`, `.txt`, `.rst`, or `.pdf` files into `docs/`, then re-run
`python ingest.py`. For PDFs: `pip install pypdf`.

## Experiments to actually learn this

Change one thing at a time and observe the effect — this is where the concepts
click:

1. **Chunking.** In `doc_chat/config.py`, set `CHUNK_SIZE = 200` (tiny) and
   re-ingest. Retrieval gets noisier — chunks lose context. Set it to 3000 and
   chunks get diluted. Find the sweet spot.
2. **Overlap.** Set `CHUNK_OVERLAP = 0`, then ask a question whose answer sits
   on a chunk boundary. Watch it fail, then turn overlap back on.
3. **Top-k.** Run `python chat.py --top-k 1` vs `--top-k 8` and see how the
   answer changes.
4. **See the geometry.** Run this — unrelated vs related text, compared:

   ```python
   from doc_chat.embedder import Embedder
   e = Embedder()
   v = e.embed(["dog", "puppy", "interest rate"])
   print(v @ v.T)   # dog·puppy is HIGH; dog·interest-rate is ~0
   ```

Everything here scales up: swap `doc_chat/store.py` for FAISS or Chroma, swap
fastembed for a bigger embedding model, and the pipeline shape stays identical.
