"""
Claim-level verification against retrieved evidence.

Preferred production path: LLM-as-judge (OpenAI-compatible API).
Fallback: similarity in the retriever's vector space (bootstrap / offline).
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence

from truthscore.types import ClaimLabel, ClaimRecord


class ClaimVerifier(Protocol):
    def verify(
        self,
        claim: str,
        evidence: Sequence[Dict[str, Any]],
        *,
        question: str = "",
    ) -> ClaimRecord:
        ...


def _merge_evidence_text(evidence: Sequence[Dict[str, Any]]) -> str:
    chunks = []
    for doc in evidence:
        t = doc.get("text") or doc.get("content") or ""
        if t:
            chunks.append(str(t))
    return "\n\n".join(chunks)


class OpenAIClaimVerifier:
    """
    LLM-as-judge: returns SUPPORTED | CONTRADICTED | UNSUPPORTED + confidence.

    Requires ``openai`` and ``OPENAI_API_KEY``. Optional: ``OPENAI_BASE_URL``,
    ``TRUTHSCORE_MODEL`` (default gpt-4o-mini).
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "Install openai: pip install 'truthscore-llm[judge]'"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAIClaimVerifier requires OPENAI_API_KEY or api_key=")
        kwargs: Dict[str, Any] = {"api_key": key}
        if base_url or os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(**kwargs)
        self._model = model or os.environ.get("TRUTHSCORE_MODEL", "gpt-4o-mini")

    def verify(
        self,
        claim: str,
        evidence: Sequence[Dict[str, Any]],
        *,
        question: str = "",
    ) -> ClaimRecord:
        ev_text = _merge_evidence_text(evidence)
        if not ev_text.strip():
            return ClaimRecord(
                text=claim,
                label=ClaimLabel.UNSUPPORTED,
                confidence=0.2,
                evidence=list(evidence),
                rationale="No evidence passages retrieved.",
            )

        schema = (
            'Respond with JSON only: {"label":"SUPPORTED|CONTRADICTED|UNSUPPORTED",'
            '"confidence":0.0-1.0,"rationale":"one sentence"}'
        )
        user = (
            f"Question (context): {question}\n\n"
            f"Claim to verify: {claim}\n\n"
            f"Evidence passages:\n{ev_text}\n\n"
            f"{schema}"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You verify whether evidence supports, contradicts, or fails to "
                        "support a factual claim. Be conservative: unsupported if evidence "
                        "does not clearly entail the claim."
                    ),
                },
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ClaimRecord(
                text=claim,
                label=ClaimLabel.UNSUPPORTED,
                confidence=0.25,
                evidence=list(evidence),
                rationale="Judge returned non-JSON.",
            )
        label_s = str(data.get("label", "UNSUPPORTED")).upper()
        try:
            label = ClaimLabel(label_s)
        except ValueError:
            label = ClaimLabel.UNSUPPORTED
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        rationale = str(data.get("rationale", ""))
        return ClaimRecord(
            text=claim,
            label=label,
            confidence=conf,
            evidence=list(evidence),
            rationale=rationale,
        )


class SimilarityEvidenceVerifier:
    """
    Bootstrap verifier: retriever-space cosine plus cheap *structural* checks
    (missing long claim tokens in evidence; negation in evidence not reflected
    in claim). This is still not NLI — use ``OpenAIClaimVerifier`` in production.
    """

    def __init__(self, vectorizer: Any) -> None:
        """
        vectorizer must implement ``similarity(query: str, doc_text: str) -> float``.
        """
        self._vectorizer = vectorizer

    @staticmethod
    def _merged_lower(evidence: Sequence[Dict[str, Any]]) -> str:
        return _merge_evidence_text(evidence).lower()

    @staticmethod
    def _top_evidence_blob(evidence: Sequence[Dict[str, Any]], k: int = 4) -> str:
        parts = []
        for doc in evidence[:k]:
            t = doc.get("text") or ""
            if t:
                parts.append(t)
        return " ".join(parts).lower()

    @staticmethod
    def _missing_content_tokens(claim: str, ev_lower: str) -> bool:
        toks = [t for t in re.findall(r"[a-z0-9]+", claim.lower()) if len(t) >= 5]
        if not toks:
            return False
        present = sum(1 for t in toks if t in ev_lower)
        return present / len(toks) < 0.72

    @staticmethod
    def _evidence_negation_unmirrored(claim_lower: str, ev_lower: str) -> bool:
        if "not" not in ev_lower and "n't" not in ev_lower and " no " not in ev_lower:
            return False
        if "not" in claim_lower or "n't" in claim_lower:
            return False
        return True

    def verify(
        self,
        claim: str,
        evidence: Sequence[Dict[str, Any]],
        *,
        question: str = "",
    ) -> ClaimRecord:
        if not evidence:
            return ClaimRecord(
                text=claim,
                label=ClaimLabel.UNSUPPORTED,
                confidence=0.15,
                evidence=[],
                rationale="No evidence retrieved.",
            )
        best = 0.0
        best_doc: Dict[str, Any] = {}
        for doc in evidence:
            t = doc.get("text") or ""
            s = float(self._vectorizer.similarity(claim, t))
            if s > best:
                best = s
                best_doc = dict(doc)

        best_text = ((best_doc.get("text") or "") if best_doc else "").lower()
        merged_lower = self._top_evidence_blob(evidence, k=4)
        claim_lower = claim.lower()
        missing = self._missing_content_tokens(claim, merged_lower)
        neg_unmirrored = self._evidence_negation_unmirrored(claim_lower, best_text)

        if missing:
            label = ClaimLabel.UNSUPPORTED
            conf = max(0.25, 0.55 - best * 0.2)
            rationale = "Key claim content is not covered by retrieved passages."
        elif neg_unmirrored and best >= 0.34:
            label = ClaimLabel.UNSUPPORTED
            conf = max(0.3, 1.0 - best * 0.35)
            rationale = "Evidence contains negation not reflected in the claim (cannot treat as support)."
        elif best >= 0.52:
            label = ClaimLabel.SUPPORTED
            conf = min(1.0, 0.52 + best * 0.48)
            rationale = "Strong overlap with retrieved passages under structural checks."
        elif best <= 0.14:
            label = ClaimLabel.UNSUPPORTED
            conf = max(0.2, 1.0 - best * 3)
            rationale = "Retrieved passages do not align with the claim."
        else:
            label = ClaimLabel.UNSUPPORTED
            conf = 0.42
            rationale = "Weak or ambiguous alignment with evidence."

        return ClaimRecord(
            text=claim,
            label=label,
            confidence=conf,
            evidence=[best_doc] if best_doc else list(evidence),
            rationale=rationale,
        )


class CallableClaimVerifier:
    """Wrap a function for tests / custom pipelines."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def verify(
        self,
        claim: str,
        evidence: Sequence[Dict[str, Any]],
        *,
        question: str = "",
    ) -> ClaimRecord:
        return self._fn(claim, evidence, question=question)
