"""
Claim-grounded truth scoring: retrieve → verify per claim → aggregate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from truthscore.claim_extractor import extract_claims_sentence
from truthscore.claim_consistency import multi_sample_claim_consistency
from truthscore.claim_verifier import ClaimVerifier, SimilarityEvidenceVerifier
from truthscore.config import DEFAULT_CONFIG, TruthScoreConfig
from truthscore.coverage import compute_coverage_score
from truthscore.default_corpus import DEFAULT_PASSAGES
from truthscore.linguistic_risk import compute_linguistic_risk
from truthscore.retrieve import EvidenceRetriever, TfidfPassageRetriever
from truthscore.types import ClaimLabel, ClaimRecord


def _claim_vote(label: ClaimLabel) -> float:
    if label == ClaimLabel.SUPPORTED:
        return 1.0
    if label == ClaimLabel.CONTRADICTED:
        return -1.0
    return -0.5


class TruthScorer:
    """
    Evaluate an answer by decomposing it into claims, retrieving evidence per
    claim, verifying each claim, then aggregating with explicit risk signals.
    """

    def __init__(
        self,
        config: Optional[TruthScoreConfig] = None,
        *,
        retriever: Optional[EvidenceRetriever] = None,
        verifier: Optional[ClaimVerifier] = None,
        claim_extractor: Optional[Callable[[str, str], List[str]]] = None,
        sample_generator: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config if config is not None else DEFAULT_CONFIG
        self.config.validate()

        self._retriever: EvidenceRetriever = retriever or TfidfPassageRetriever(
            DEFAULT_PASSAGES
        )
        self._verifier: ClaimVerifier
        if verifier is not None:
            self._verifier = verifier
        else:
            if not hasattr(self._retriever, "similarity"):
                raise TypeError(
                    "Default SimilarityEvidenceVerifier requires a retriever with "
                    "similarity(a, b), e.g. TfidfPassageRetriever."
                )
            self._verifier = SimilarityEvidenceVerifier(self._retriever)
        self._claim_extractor = claim_extractor or extract_claims_sentence
        self._sample_generator = sample_generator

    def _make_decision(self, truth_score: float) -> str:
        if truth_score >= self.config.accept_threshold:
            return "ACCEPT"
        if truth_score >= self.config.qualified_threshold:
            return "QUALIFIED"
        return "REFUSE"

    def score(
        self,
        question: str,
        answer: str,
        evidence: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Score ``answer`` against ``question``.

        If ``evidence`` is provided, it must be a list of dicts with ``text`` keys;
        that list is reused for every claim (caller-controlled grounding). When
        omitted, retrieval runs **per claim**.
        """
        try:
            claims = self._claim_extractor(
                question, answer, self.config.claim_min_words
            )
        except TypeError:
            claims = self._claim_extractor(question, answer)
        if not claims:
            return self._empty_result(question, answer)

        claim_records: List[ClaimRecord] = []
        for claim in claims:
            if evidence is not None:
                docs = list(evidence)
            else:
                docs = self._retriever.retrieve(claim, self.config.top_k)
            rec = self._verifier.verify(claim, docs, question=question)
            # Preserve full retrieval set for coverage / audit
            rec.evidence = docs
            claim_records.append(rec)

        votes = [_claim_vote(r.label) for r in claim_records]
        mean_raw = sum(votes) / len(votes)
        base_01 = (mean_raw + 1.0) / 2.0

        unsupported = sum(1 for r in claim_records if r.label == ClaimLabel.UNSUPPORTED)
        contradictions = sum(1 for r in claim_records if r.label == ClaimLabel.CONTRADICTED)
        unsupported_ratio = unsupported / len(claim_records)

        truth_score = base_01
        truth_score -= self.config.penalty_unsupported_ratio * unsupported_ratio
        if contradictions > 0:
            truth_score -= self.config.penalty_contradiction

        risk = compute_linguistic_risk(answer, claim_records)
        truth_score -= self.config.penalty_overclaim * risk.overclaim_risk
        truth_score = max(0.0, min(1.0, truth_score))

        consistency_score = 1.0
        multi_sample_used = False
        if self._sample_generator is not None:
            consistency_score, _ = multi_sample_claim_consistency(
                question,
                self._sample_generator,
                n_samples=self.config.claim_consistency_samples,
                min_words=self.config.claim_min_words,
            )
            multi_sample_used = True
            w = self.config.consistency_blend_weight
            truth_score = max(0.0, min(1.0, (1 - w) * truth_score + w * consistency_score))

        coverage = compute_coverage_score(question, answer, claim_records)
        mean_conf = sum(r.confidence for r in claim_records) / len(claim_records)

        decision = self._make_decision(truth_score)

        return {
            "truth_score": truth_score,
            "decision": decision,
            "claims": [r.to_dict() for r in claim_records],
            "unsupported_ratio": unsupported_ratio,
            "contradictions": contradictions,
            "consistency_score": consistency_score,
            "multi_sample_used": multi_sample_used,
            "linguistic_risk": risk.overclaim_risk,
            "linguistic_breakdown": risk.to_dict(),
            "emotional_intensity": risk.emotional_intensity,
            "coverage": coverage,
            "evidence_score": mean_conf,
            "claim_raw_mean": mean_raw,
            # Back-compat aliases (deprecated)
            "consistency": consistency_score,
            "language_confidence": 1.0 - risk.overclaim_risk,
        }

    def _empty_result(self, question: str, answer: str) -> Dict[str, Any]:
        risk = compute_linguistic_risk(answer or "", [])
        return {
            "truth_score": 0.0,
            "decision": "REFUSE",
            "claims": [],
            "unsupported_ratio": 1.0,
            "contradictions": 0,
            "consistency_score": 1.0,
            "multi_sample_used": False,
            "linguistic_risk": risk.overclaim_risk,
            "linguistic_breakdown": risk.to_dict(),
            "emotional_intensity": risk.emotional_intensity,
            "coverage": 0.0,
            "evidence_score": 0.0,
            "claim_raw_mean": 0.0,
            "consistency": 1.0,
            "language_confidence": max(0.0, 1.0 - risk.overclaim_risk),
        }
