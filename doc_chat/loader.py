"""Read documents from docs/ into plain text."""
from __future__ import annotations

from pathlib import Path

from .config import DOCS_DIR, PDF_SUFFIXES, TEXT_SUFFIXES


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Reading {path.name} needs pypdf. Install it with:  pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def iter_docs(docs_dir: Path = DOCS_DIR) -> list[tuple[str, str]]:
    """Return a list of (source_name, text) for every supported file under docs/."""
    docs: list[tuple[str, str]] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix in PDF_SUFFIXES:
            text = _read_pdf(path)
        else:
            continue
        if text.strip():
            docs.append((str(path.relative_to(docs_dir)), text))
    return docs
