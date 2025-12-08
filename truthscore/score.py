"""
Main scoring module implementing the TruthScorer class.

This module provides the primary interface for computing truth scores
for LLM-generated answers based on evidence agreement, consistency,
and coverage metrics.
"""

import math
from typing import Dict, Optional

from truthscore.config import TruthScoreConfig, DEFAULT_CONFIG
from truthscore.retrieve import retrieve_evidence
from truthscore.nli import compute_evidence_score
from truthscore.consistency import estimate_consistency, compute_language_confidence
from truthscore.coverage import compute_coverage_score


class TruthScorer:
    """
    Main class for computing truth scores of LLM-generated answers.
    
    The TruthScorer evaluates answers based on multiple dimensions:
    - Evidence agreement: How well retrieved evidence supports the answer
    - Self-consistency: Internal coherence of the answer
    - Retrieval coverage: Comprehensiveness of supporting evidence
    - Language confidence: Linguistic quality and certainty indicators
    
    Example:
        >>> scorer = TruthScorer()
        >>> result = scorer.score(
        ...     question="What is the capital of France?",
        ...     answer="The capital of France is Paris."
        ... )
        >>> print(result['truth_score'])
        0.85
    """
    
    def __init__(self, config: Optional[TruthScoreConfig] = None):
        """
        Initialize TruthScorer with configuration.
        
        Args:
            config: TruthScoreConfig instance. If None, uses DEFAULT_CONFIG.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        self.config.validate()
    
    def _sigmoid_normalize(self, score: float) -> float:
        """
        Apply sigmoid normalization to a score.
        
        Maps a score to [0.0, 1.0] using a sigmoid function, which
        provides smooth transitions and can help with thresholding.
        
        Args:
            score: Raw score to normalize (typically in [0.0, 1.0]).
        
        Returns:
            Normalized score in [0.0, 1.0].
        """
        # Shift and scale to center around sigmoid_center
        shifted = score - self.config.sigmoid_center
        
        # Apply sigmoid: 1 / (1 + exp(-k * x))
        # where k is steepness and x is the shifted score
        sigmoid = 1.0 / (1.0 + math.exp(-self.config.sigmoid_steepness * shifted))
        
        return sigmoid
    
    def _aggregate_scores(
        self,
        evidence_score: float,
        consistency: float,
        coverage: float,
        language_confidence: float
    ) -> float:
        """
        Aggregate component scores into a single truth score.
        
        Uses weighted combination of component scores with optional
        sigmoid normalization.
        
        Args:
            evidence_score: Evidence agreement score [0.0, 1.0]
            consistency: Self-consistency score [0.0, 1.0]
            coverage: Retrieval coverage score [0.0, 1.0]
            language_confidence: Language confidence score [0.0, 1.0]
        
        Returns:
            Aggregated truth score in [0.0, 1.0].
        """
        # Weighted linear combination
        raw_score = (
            evidence_score * self.config.evidence_weight +
            consistency * self.config.consistency_weight +
            coverage * self.config.coverage_weight +
            language_confidence * self.config.language_weight
        )
        
        # Apply sigmoid normalization
        normalized_score = self._sigmoid_normalize(raw_score)
        
        return max(0.0, min(1.0, normalized_score))
    
    def _make_decision(self, truth_score: float) -> str:
        """
        Make acceptance decision based on truth score and thresholds.
        
        Decision logic:
        - ACCEPT: truth_score >= accept_threshold
        - QUALIFIED: qualified_threshold <= truth_score < accept_threshold
        - REFUSE: truth_score < qualified_threshold
        
        Args:
            truth_score: Computed truth score [0.0, 1.0]
        
        Returns:
            Decision string: "ACCEPT", "QUALIFIED", or "REFUSE"
        """
        if truth_score >= self.config.accept_threshold:
            return "ACCEPT"
        elif truth_score >= self.config.qualified_threshold:
            return "QUALIFIED"
        else:
            return "REFUSE"
    
    def score(
        self,
        question: str,
        answer: str,
        evidence: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Compute truth score for a question-answer pair.
        
        This is the main entry point for truth scoring. It orchestrates
        evidence retrieval, entailment checking, consistency evaluation,
        and coverage assessment to produce a comprehensive truth score.
        
        Args:
            question: The question being answered.
            answer: The LLM-generated answer to evaluate.
            evidence: Optional pre-retrieved evidence documents. If None,
                     evidence will be retrieved automatically.
        
        Returns:
            Dictionary containing:
            - 'truth_score': Overall truth score [0.0, 1.0]
            - 'decision': Decision string ("ACCEPT", "QUALIFIED", "REFUSE")
            - 'evidence_score': Evidence agreement score [0.0, 1.0]
            - 'consistency': Self-consistency score [0.0, 1.0]
            - 'language_confidence': Language confidence score [0.0, 1.0]
            - 'coverage': Retrieval coverage score [0.0, 1.0]
        
        Example:
            >>> scorer = TruthScorer()
            >>> result = scorer.score(
            ...     question="Does vitamin C prevent the common cold?",
            ...     answer="Vitamin C prevents the common cold."
            ... )
            >>> print(result['truth_score'])
            0.72
        """
        # Retrieve evidence if not provided
        if evidence is None:
            evidence = retrieve_evidence(question, answer)
        
        # Compute component scores
        evidence_score = compute_evidence_score(answer, evidence)
        consistency = estimate_consistency(question, answer)
        language_confidence = compute_language_confidence(answer)
        coverage = compute_coverage_score(question, answer, evidence)
        
        # Aggregate into final truth score
        truth_score = self._aggregate_scores(
            evidence_score,
            consistency,
            coverage,
            language_confidence
        )
        
        # Make decision
        decision = self._make_decision(truth_score)
        
        return {
            'truth_score': truth_score,
            'decision': decision,
            'evidence_score': evidence_score,
            'consistency': consistency,
            'language_confidence': language_confidence,
            'coverage': coverage
        }

