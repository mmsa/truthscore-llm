"""
TruthScore: A research library for evaluating truthfulness of LLM outputs.

This library implements a "Truth Score" system that evaluates LLM-generated
answers based on evidence agreement, self-consistency, retrieval coverage,
and language confidence metrics.

Main API:
    from truthscore import TruthScorer
    
    scorer = TruthScorer()
    result = scorer.score(
        question="Your question here",
        answer="LLM-generated answer"
    )

Research Disclaimer:
    This library is provided for research purposes. The scoring mechanisms
    are experimental and should not be used as the sole basis for critical
    decisions without validation and domain-specific calibration.
"""

from truthscore.score import TruthScorer
from truthscore.config import TruthScoreConfig, DEFAULT_CONFIG

__version__ = "0.1.2"
__all__ = ['TruthScorer', 'TruthScoreConfig', 'DEFAULT_CONFIG']

