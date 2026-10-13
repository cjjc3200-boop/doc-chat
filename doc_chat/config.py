"""Central config. Tweak the numbers here to see how they change retrieval."""
from pathlib import Path

# Where the docs live and where the index is written
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
INDEX_PATH = ROOT / "index" / "store"  # -> store.npy + store.json

# Embedding model (downloaded once, ~130MB, cached locally)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Chunking knobs (these are the levers you'll experiment with) ---
CHUNK_SIZE = 700      # target characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbours (keeps context across the seam)

# --- Retrieval ---
TOP_K = 4             # how many passages to feed the answer step

# Supported file types
TEXT_SUFFIXES = {".md", ".txt", ".rst"}
PDF_SUFFIXES = {".pdf"}
