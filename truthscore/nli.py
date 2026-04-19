"""
Deprecated module: claim verification moved to ``truthscore.claim_verifier``.

Kept for backwards imports of ``check_entailment`` in older scripts.
"""

from __future__ import annotations

import warnings
from typing import Dict, List


def check_entailment(claim: str, evidence_text: str) -> Dict[str, float]:
    warnings.warn(
        "truthscore.nli.check_entailment is deprecated; use claim_verifier instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from truthscore.retrieve import TfidfPassageRetriever
    from truthscore.default_corpus import DEFAULT_PASSAGES

    r = TfidfPassageRetriever(DEFAULT_PASSAGES)
    sim = r.similarity(claim, evidence_text)
    entailment = min(1.0, max(0.0, sim))
    contradiction = max(0.0, min(1.0, (1.0 - sim) * 0.35))
    neutral = max(0.0, 1.0 - entailment - contradiction)
    tot = entailment + contradiction + neutral
    if tot > 0:
        entailment /= tot
        contradiction /= tot
        neutral /= tot
    return {"entailment": entailment, "contradiction": contradiction, "neutral": neutral}


def compute_evidence_score(claim: str, evidence_documents: List[Dict[str, str]]) -> float:
    warnings.warn(
        "truthscore.nli.compute_evidence_score is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not evidence_documents:
        return 0.0
    acc = 0.0
    for doc in evidence_documents:
        t = doc.get("text", "")
        if not t:
            continue
        r = check_entailment(claim, t)
        acc += r["entailment"] - r["contradiction"]
    v = acc / len(evidence_documents)
    return max(0.0, min(1.0, (v + 1.0) / 2.0))
