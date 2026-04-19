"""
Configuration for claim-grounded TruthScorer.
"""

from dataclasses import dataclass


@dataclass
class TruthScoreConfig:
    """Thresholds and penalty weights for claim-level aggregation."""

    accept_threshold: float = 0.72
    qualified_threshold: float = 0.52

    top_k: int = 5
    claim_min_words: int = 3

    # Penalties applied after mapping raw claim vote mean to [0, 1]
    penalty_unsupported_ratio: float = 0.22
    penalty_contradiction: float = 0.18
    penalty_overclaim: float = 0.14

    # When ``sample_generator`` is set, blend claim-set stability slightly
    consistency_blend_weight: float = 0.08
    claim_consistency_samples: int = 5

    def validate(self) -> None:
        if not (0.0 <= self.accept_threshold <= 1.0):
            raise ValueError("accept_threshold must be in [0, 1]")
        if not (0.0 <= self.qualified_threshold <= 1.0):
            raise ValueError("qualified_threshold must be in [0, 1]")
        if self.accept_threshold <= self.qualified_threshold:
            raise ValueError("accept_threshold must be > qualified_threshold")
        for name in (
            "penalty_unsupported_ratio",
            "penalty_contradiction",
            "penalty_overclaim",
            "consistency_blend_weight",
        ):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


DEFAULT_CONFIG = TruthScoreConfig()
