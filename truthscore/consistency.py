"""
Claim-level multi-sample consistency lives in ``claim_consistency``.

Legacy surface-level consistency helpers were removed; use
``TruthScorer(..., sample_generator=...)`` for stability across samples.
"""

from truthscore.claim_consistency import (
    claim_set_signature,
    multi_sample_claim_consistency,
)

__all__ = [
    "multi_sample_claim_consistency",
    "claim_set_signature",
]
