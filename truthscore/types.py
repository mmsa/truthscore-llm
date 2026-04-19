"""Shared types for claim-level truth scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ClaimLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ClaimRecord:
    """One atomic claim and its verification outcome."""

    text: str
    label: ClaimLabel
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "rationale": self.rationale,
        }


@dataclass
class LinguisticRiskReport:
    """Style / rhetoric signals — not truth."""

    overclaim_risk: float
    emotional_intensity: float
    assertive_tone: float
    weak_evidence_mass: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overclaim_risk": self.overclaim_risk,
            "emotional_intensity": self.emotional_intensity,
            "assertive_tone": self.assertive_tone,
            "weak_evidence_mass": self.weak_evidence_mass,
        }
