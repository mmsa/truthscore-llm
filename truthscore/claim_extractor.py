"""
Split answers into atomic factual claims for per-claim verification.

Default path is deterministic (sentence / clause splitting). Optional LLM
extraction can be wired in later via a custom callable.
"""

from __future__ import annotations

import re
from typing import Callable, List, Sequence


def _normalize_sentence(s: str) -> str:
    s = " ".join(s.split())
    return s.strip(" \t\n\r\"'")


def extract_claims_sentence(_question: str, answer: str, min_words: int = 3) -> List[str]:
    """
    Heuristic claim extraction: split on sentence boundaries and conjunctions.

    This is generic (no domain lexicon). For production, pass an LLM-backed
    extractor to TruthScorer(claim_extractor=...).
    """
    if not answer or not answer.strip():
        return []

    text = answer.strip()
    # Split numbered/bullet lines into separate candidates
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?:\s+and\s+)", text, flags=re.IGNORECASE)
    claims: List[str] = []
    seen = set()
    for p in parts:
        c = _normalize_sentence(p)
        if not c or len(c) < 8:
            continue
        words = len(c.split())
        if words < min_words:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(c)
    if not claims and text:
        if len(text.split()) >= min_words:
            claims.append(_normalize_sentence(text))
    return claims


ClaimExtractorFn = Callable[[str, str], List[str]]
