"""
Linguistic / rhetorical risk — NOT truth.

``overclaim_risk`` rises when the prose is assertive or emotional while
evidence support (from claim verification) is weak.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from truthscore.types import ClaimLabel, ClaimRecord, LinguisticRiskReport


def emotional_intensity(text: str) -> float:
    """
    Cheap, lexicon-free emotional proxy: exclamation density + ALL-CAPS tokens.

    This is NOT EmoTFIDF; if you install an emotive lexicon model, replace this
    function and keep the same name for drop-in use in TruthScorer.
    """
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    n = max(1, len(t))
    bang = t.count("!") / max(8, n / 40)
    toks = re.findall(r"[A-Za-z]+", t)
    if not toks:
        return float(min(1.0, bang))
    caps = sum(1 for w in toks if len(w) > 1 and w.isupper()) / len(toks)
    score = min(1.0, 0.55 * min(1.0, bang * 3) + 0.45 * min(1.0, caps * 4))
    return score


def assertive_tone_score(text: str) -> float:
    """Structural assertiveness: short sentences, many periods, few modals-as-?"""
    if not text or not text.strip():
        return 0.0
    words = text.split()
    n = max(1, len(words))
    periods = text.count(".") + text.count("!")
    density = periods / max(1, n / 12)
    # Many short punchy clauses read as assertive in product risk scoring
    return float(min(1.0, density * 0.85))


def weak_evidence_mass(claim_records: Sequence[ClaimRecord]) -> float:
    """Fraction of claims that are not SUPPORTED with decent confidence."""
    if not claim_records:
        return 1.0
    bad = 0
    for r in claim_records:
        if r.label != ClaimLabel.SUPPORTED:
            bad += 1
        elif r.confidence < 0.55:
            bad += 0.5
    return min(1.0, bad / len(claim_records))


def compute_linguistic_risk(
    answer: str,
    claim_records: Sequence[ClaimRecord],
) -> LinguisticRiskReport:
    emo = emotional_intensity(answer)
    assertive = assertive_tone_score(answer)
    weak = weak_evidence_mass(claim_records)

    # Overclaim: assertive / emotional voice while evidence is thin
    base = 0.45 * assertive + 0.35 * weak + 0.2 * emo
    if emo > 0.35 and weak > 0.45:
        base = min(1.0, base + 0.18)
    overclaim = float(max(0.0, min(1.0, base)))

    return LinguisticRiskReport(
        overclaim_risk=overclaim,
        emotional_intensity=emo,
        assertive_tone=assertive,
        weak_evidence_mass=weak,
    )
