"""
TruthScore: claim-grounded evaluation of LLM answers with explicit risk signals.
"""

from truthscore.score import TruthScorer
from truthscore.config import TruthScoreConfig
from truthscore.types import ClaimLabel, ClaimRecord, LinguisticRiskReport
from truthscore.claim_verifier import (
    CallableClaimVerifier,
    OpenAIClaimVerifier,
    SimilarityEvidenceVerifier,
)
from truthscore.retrieve import TfidfPassageRetriever, build_faiss_retriever
from truthscore.production import create_production_scorer
from truthscore.io_corpus import load_passages_from_file
from truthscore.wikipedia_retriever import WikipediaRetriever

__version__ = "0.2.0"
__all__ = [
    "TruthScorer",
    "TruthScoreConfig",
    "ClaimLabel",
    "ClaimRecord",
    "LinguisticRiskReport",
    "CallableClaimVerifier",
    "OpenAIClaimVerifier",
    "SimilarityEvidenceVerifier",
    "TfidfPassageRetriever",
    "build_faiss_retriever",
    "create_production_scorer",
    "load_passages_from_file",
    "WikipediaRetriever",
]
