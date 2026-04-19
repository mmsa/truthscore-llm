"""
Load passage lists from disk for ``TfidfPassageRetriever`` / FAISS indexing.

Supported formats:

- **.jsonl**: one JSON object per line. Each object should include a ``"text"``
  field (or ``"content"`` / ``"body"``). Optional ``"source"`` is preserved when
  building retriever documents (TF-IDF only uses ``text`` for vectors).
- **.txt**: one passage per non-empty line, or paragraphs separated by a blank line.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import List


def load_passages_from_file(path: str | Path) -> List[str]:
    """
    Return plain passage strings for indexing.

    For JSONL with metadata you only need passages here; use your own loader
    if you must keep per-document metadata alongside TF-IDF.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Corpus file not found: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".jsonl":
        return _load_jsonl(text)
    return _load_plaintext(text)


def _load_jsonl(raw: str) -> List[str]:
    out: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, str):
            chunk = obj.strip()
        elif isinstance(obj, dict):
            chunk = (
                obj.get("text")
                or obj.get("content")
                or obj.get("body")
                or ""
            )
            if not isinstance(chunk, str):
                chunk = ""
            chunk = chunk.strip()
        else:
            continue
        if len(chunk) >= 8:
            out.append(chunk)
    if not out:
        raise ValueError("JSONL corpus contained no usable text fields.")
    return out


def _load_plaintext(raw: str) -> List[str]:
    if "\n\n" in raw:
        parts = re.split(r"\n\s*\n", raw.strip())
    else:
        parts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out = [html.unescape(p.strip()) for p in parts if len(p.strip()) >= 8]
    if not out:
        raise ValueError("Plain-text corpus contained no passages of sufficient length.")
    return out
