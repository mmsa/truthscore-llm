"""
Production wiring: real evidence retrieval + configurable claim judge.

**Default (no vendor lock-in):** Wikipedia or file corpus retrieval with
``SimilarityEvidenceVerifier`` (lexical / structural checks over retrieved text).

**Optional LLM judge:** set ``judge="openai"`` (or ``TRUTHSCORE_JUDGE=openai``) and
provide ``OPENAI_API_KEY`` if you explicitly want OpenAI-compatible chat judging.
Alternatively pass any ``verifier=`` implementing ``ClaimVerifier``.

Environment variables:

- ``TRUTHSCORE_EVIDENCE_MODE`` — ``wikipedia`` (default) or ``corpus``.
- ``TRUTHSCORE_CORPUS_PATH`` — path to ``.jsonl`` or ``.txt`` when mode is ``corpus``.
- ``TRUTHSCORE_JUDGE`` — ``similarity`` (default) or ``openai``.
- ``TRUTHSCORE_WIKIPEDIA_LANG`` — wiki language code (default ``en``).
- ``TRUTHSCORE_USER_AGENT`` — recommended for Wikipedia API traffic.

OpenAI-only when ``judge="openai"``:

- ``OPENAI_API_KEY``, optional ``OPENAI_BASE_URL``, ``TRUTHSCORE_MODEL``.
- Install: ``pip install 'truthscore-llm[judge]'``.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Literal, Optional

from truthscore.claim_verifier import ClaimVerifier, OpenAIClaimVerifier, SimilarityEvidenceVerifier
from truthscore.config import TruthScoreConfig
from truthscore.io_corpus import load_passages_from_file
from truthscore.retrieve import TfidfPassageRetriever
from truthscore.score import TruthScorer
from truthscore.wikipedia_retriever import WikipediaRetriever

logger = logging.getLogger(__name__)

EvidenceMode = Literal["wikipedia", "corpus"]
def _build_verifier(
    retriever,
    *,
    verifier: Optional[ClaimVerifier],
    judge: Optional[str],
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    model: Optional[str],
) -> ClaimVerifier:
    if verifier is not None:
        return verifier

    raw = (judge or os.environ.get("TRUTHSCORE_JUDGE") or "similarity").lower().strip()
    if raw == "openai":
        key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "judge='openai' requires OPENAI_API_KEY or openai_api_key= "
                "and: pip install 'truthscore-llm[judge]'."
            )
        return OpenAIClaimVerifier(
            api_key=key,
            base_url=openai_base_url or os.environ.get("OPENAI_BASE_URL"),
            model=model or os.environ.get("TRUTHSCORE_MODEL"),
        )
    if raw != "similarity":
        raise ValueError(
            f"Unknown judge mode {raw!r}. Use 'similarity', 'openai', or pass verifier=."
        )
    return SimilarityEvidenceVerifier(retriever)


def create_production_scorer(
    *,
    config: Optional[TruthScoreConfig] = None,
    evidence_mode: Optional[EvidenceMode] = None,
    corpus_path: Optional[str] = None,
    verifier: Optional[ClaimVerifier] = None,
    judge: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    model: Optional[str] = None,
    wikipedia_lang: Optional[str] = None,
    wikipedia_user_agent: Optional[str] = None,
    wikipedia_timeout_s: float = 15.0,
    sample_generator: Optional[Callable[[str], str]] = None,
) -> TruthScorer:
    """
    Build a ``TruthScorer`` with Wikipedia or file-backed retrieval.

    Pass ``verifier=`` for your own model (local HF, Anthropic, HTTP, etc.), or use
    ``judge="similarity"`` (default) / ``judge="openai"`` for built-in options.
    """
    mode_raw = (evidence_mode or os.environ.get("TRUTHSCORE_EVIDENCE_MODE") or "wikipedia").lower()
    if mode_raw not in ("wikipedia", "corpus"):
        raise ValueError(
            f"evidence_mode must be 'wikipedia' or 'corpus', got {mode_raw!r}. "
            "Set TRUTHSCORE_EVIDENCE_MODE or pass evidence_mode=."
        )
    mode: EvidenceMode = mode_raw  # type: ignore[assignment]

    corpus_passages = None
    corpus_file: Optional[str] = None
    if mode == "corpus":
        corpus_file = corpus_path or os.environ.get("TRUTHSCORE_CORPUS_PATH")
        if not corpus_file:
            raise ValueError(
                "evidence_mode='corpus' requires a corpus file. "
                "Set TRUTHSCORE_CORPUS_PATH or pass corpus_path=."
            )
        corpus_passages = load_passages_from_file(corpus_file)
    else:
        extra = corpus_path or os.environ.get("TRUTHSCORE_CORPUS_PATH")
        if extra:
            logger.info("TRUTHSCORE_CORPUS_PATH is set but evidence_mode is wikipedia; corpus path ignored.")

    if mode == "wikipedia":
        lang = wikipedia_lang or os.environ.get("TRUTHSCORE_WIKIPEDIA_LANG", "en")
        retriever = WikipediaRetriever(
            lang=lang,
            user_agent=wikipedia_user_agent,
            timeout_s=wikipedia_timeout_s,
        )
        logger.info("Production TruthScorer: Wikipedia (%s).", lang)
    else:
        if corpus_passages is None or corpus_file is None:
            raise RuntimeError("Corpus passages not loaded.")
        retriever = TfidfPassageRetriever(corpus_passages, source_prefix="corpus")
        logger.info(
            "Production TruthScorer: TF-IDF over %s passages from %s.",
            len(corpus_passages),
            corpus_file,
        )

    v = _build_verifier(
        retriever,
        verifier=verifier,
        judge=judge,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        model=model,
    )
    jname = type(v).__name__
    logger.info("Production claim judge: %s", jname)

    return TruthScorer(
        config=config,
        retriever=retriever,
        verifier=v,
        sample_generator=sample_generator,
    )
