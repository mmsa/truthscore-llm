"""
Configuration module for TruthScore library.

This module defines default parameters, thresholds, and weights used
throughout the truth scoring system.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TruthScoreConfig:
    """
    Configuration class for TruthScorer.
    
    Attributes:
        evidence_weight: Weight for evidence agreement score (0.0-1.0)
        consistency_weight: Weight for self-consistency score (0.0-1.0)
        coverage_weight: Weight for retrieval coverage score (0.0-1.0)
        language_weight: Weight for language confidence score (0.0-1.0)
        accept_threshold: Minimum truth_score for ACCEPT decision
        qualified_threshold: Minimum truth_score for QUALIFIED decision
        sigmoid_steepness: Steepness parameter for sigmoid normalization
        sigmoid_center: Center point for sigmoid normalization
    """
    
    evidence_weight: float = 0.6
    consistency_weight: float = 0.2
    coverage_weight: float = 0.15
    language_weight: float = 0.05
    
    accept_threshold: float = 0.75
    qualified_threshold: float = 0.55
    
    sigmoid_steepness: float = 10.0
    sigmoid_center: float = 0.5
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If weights don't sum to approximately 1.0 or
                       if thresholds are invalid.
        """
        total_weight = (
            self.evidence_weight +
            self.consistency_weight +
            self.coverage_weight +
            self.language_weight
        )
        
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}"
            )
        
        if not (0.0 <= self.accept_threshold <= 1.0):
            raise ValueError(
                f"accept_threshold must be in [0.0, 1.0], got {self.accept_threshold}"
            )
        
        if not (0.0 <= self.qualified_threshold <= 1.0):
            raise ValueError(
                f"qualified_threshold must be in [0.0, 1.0], got {self.qualified_threshold}"
            )
        
        if self.accept_threshold <= self.qualified_threshold:
            raise ValueError(
                f"accept_threshold ({self.accept_threshold}) must be > "
                f"qualified_threshold ({self.qualified_threshold})"
            )


# Default configuration instance
DEFAULT_CONFIG = TruthScoreConfig()

