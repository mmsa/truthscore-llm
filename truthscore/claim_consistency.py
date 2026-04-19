"""
Multi-sample consistency at the claim level.

Provide ``sample_generator(question) -> str`` to TruthScorer; otherwise
consistency is reported as neutral (1.0) with ``multi_sample_used=False``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set, Tuple

from truthscore.claim_extractor import extract_claims_sentence


def _norm_claim(s: str) -> str:
    return " ".join(s.lower().split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def claim_set_signature(claims: List[str]) -> Set[str]:
    """Token multiset signature for stability (lightweight)."""
    out: Set[str] = set()
    for c in claims:
        toks = tuple(sorted(_norm_claim(c).split()))
        out.add("|".join(toks[:40]))
    return out


def multi_sample_claim_consistency(
    question: str,
    sample_generator: Callable[[str], str],
    *,
    n_samples: int = 5,
    min_words: int = 3,
) -> Tuple[float, List[List[str]]]:
    """
    Draw N answers, extract claims from each, measure pairwise stability.

    Returns (score in [0,1], list of claim lists per sample).
    """
    claim_lists: List[List[str]] = []
    for _ in range(max(2, n_samples)):
        ans = sample_generator(question)
        claim_lists.append(
            extract_claims_sentence(question, ans, min_words=min_words)
        )
    sigs = [claim_set_signature(cl) for cl in claim_lists]
    pairs = 0
    acc = 0.0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            pairs += 1
            acc += _jaccard(sigs[i], sigs[j])
    score = acc / pairs if pairs else 1.0
    return float(max(0.0, min(1.0, score))), claim_lists
