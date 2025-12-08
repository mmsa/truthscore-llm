"""
Coverage module for retrieval coverage evaluation.

This module provides functionality to assess how well retrieved evidence
covers the claims and topics in an answer.
"""

from typing import List, Dict


def compute_coverage_score(
    question: str,
    answer: str,
    evidence_documents: List[Dict[str, str]]
) -> float:
    """
    Compute coverage score for evidence documents.
    
    Coverage measures the extent to which retrieved evidence documents
    cover the topics, claims, and information needs expressed in the
    question and answer.
    
    Args:
        question: The original question.
        answer: The answer being evaluated.
        evidence_documents: List of retrieved evidence documents.
    
    Returns:
        Coverage score in [0.0, 1.0], where:
        - 1.0 indicates comprehensive coverage
        - 0.0 indicates insufficient or irrelevant coverage
    
    Note:
        This function delegates to retrieve.compute_retrieval_coverage
        for consistency. In production, more sophisticated coverage
        metrics could be implemented here.
    """
    from truthscore.retrieve import compute_retrieval_coverage
    
    return compute_retrieval_coverage(question, answer, evidence_documents)

