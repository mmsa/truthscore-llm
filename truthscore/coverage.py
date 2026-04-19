"""
Coverage: how well retrieved evidence matches each claim (top relevance).
"""

from typing import Any, Sequence

from truthscore.retrieve import compute_retrieval_coverage


def compute_coverage_score(
    question: str,
    answer: str,
    claim_records: Sequence[Any],
) -> float:
    """
    Args:
        question: Unused placeholder for API compatibility.
        answer: Unused placeholder for API compatibility.
        claim_records: Sequence of ``ClaimRecord`` instances from scoring.
    """
    return compute_retrieval_coverage(claim_records)
