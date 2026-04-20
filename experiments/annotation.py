"""
Annotation utilities for experiment outcomes.

Implements outcome categories for manual annotation:
- Correct Answer
- Overconfident Error
- Correct Refusal
- Hedged but Incorrect
"""

from typing import Dict, List, Optional
from enum import Enum


class OutcomeCategory(Enum):
    """Outcome categories for experiment responses."""
    CORRECT_ANSWER = "Correct Answer"
    OVERCONFIDENT_ERROR = "Overconfident Error"
    CORRECT_REFUSAL = "Correct Refusal"
    HEDGED_BUT_INCORRECT = "Hedged but Incorrect"


class Annotator:
    """Helper class for manual annotation of experiment results."""
    
    @staticmethod
    def annotate(
        prompt: str,
        answer: str,
        ground_truth: Optional[str] = None,
        is_correct: Optional[bool] = None,
        is_refusal: bool = False,
        is_hedged: bool = False
    ) -> OutcomeCategory:
        """
        Annotate a response based on provided information.
        
        Args:
            prompt: The original question/prompt
            answer: The generated answer
            ground_truth: Ground truth answer (if known)
            is_correct: Whether answer is correct (if known)
            is_refusal: Whether answer is a refusal
            is_hedged: Whether answer contains hedging language
        
        Returns:
            OutcomeCategory enum value
        """
        # Correct Refusal: Answer refuses and should refuse
        if is_refusal:
            # In real annotation, would check if refusal is appropriate
            # For now, assume refusal is correct if ground_truth indicates uncertainty
            if ground_truth is None or "unknown" in ground_truth.lower():
                return OutcomeCategory.CORRECT_REFUSAL
            # Could also be incorrect refusal, but defaulting to correct
        
        # Correct Answer: Answer is correct and confident
        if is_correct and not is_hedged and not is_refusal:
            return OutcomeCategory.CORRECT_ANSWER
        
        # Overconfident Error: Answer is wrong but confident
        if is_correct is False and not is_hedged and not is_refusal:
            return OutcomeCategory.OVERCONFIDENT_ERROR
        
        # Hedged but Incorrect: Answer is wrong but hedged
        if is_correct is False and is_hedged:
            return OutcomeCategory.HEDGED_BUT_INCORRECT
        
        # Default: if unsure, return Correct Answer (conservative)
        return OutcomeCategory.CORRECT_ANSWER
    
    @staticmethod
    def detect_hedging(answer: str) -> bool:
        """
        Detect if answer contains hedging language.
        
        Args:
            answer: The answer text
        
        Returns:
            True if answer contains hedging markers
        """
        hedging_markers = [
            "maybe", "perhaps", "possibly", "might", "could", "uncertain",
            "unclear", "unknown", "probably", "likely", "seems", "appears",
            "suggest", "indicate", "may", "not sure", "uncertain"
        ]
        
        answer_lower = answer.lower()
        return any(marker in answer_lower for marker in hedging_markers)
    
    @staticmethod
    def detect_refusal(answer: str) -> bool:
        """
        Detect if answer is a refusal.
        
        Args:
            answer: The answer text
        
        Returns:
            True if answer appears to be a refusal
        """
        refusal_markers = [
            "cannot", "cannot provide", "unable to", "don't know",
            "no information", "insufficient", "not confident",
            "cannot answer", "unable to answer"
        ]
        
        answer_lower = answer.lower()
        return any(marker in answer_lower for marker in refusal_markers)

