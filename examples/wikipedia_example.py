"""
Score an answer using live Wikipedia as evidence (MediaWiki API).

Requirements:
  - Network access to *.wikipedia.org
  - pip install truthscore-llm  (no OpenAI needed for the default similarity judge)

Wikimedia asks for a descriptive User-Agent. Set TRUTHSCORE_USER_AGENT or edit
the string below before production use:
  https://meta.wikimedia.org/wiki/User-Agent_policy

Run from repo root:
  python examples/wikipedia_example.py

Or with PYTHONPATH if not installed:
  PYTHONPATH=. python examples/wikipedia_example.py
"""

from __future__ import annotations

import os

from truthscore import create_production_scorer


def main() -> None:
    user_agent = os.environ.get(
        "TRUTHSCORE_USER_AGENT",
        "TruthScore-WikipediaExample/1.0 (+https://github.com/mmsa/truthscore-llm; research)",
    )

    # Wikipedia retrieval + SimilarityEvidenceVerifier (no cloud LLM).
    scorer = create_production_scorer(
        evidence_mode="wikipedia",
        wikipedia_lang="en",
        wikipedia_user_agent=user_agent,
        judge="similarity",
    )

    question = "What is the capital of France?"
    answer = "The capital of France is Paris."

    result = scorer.score(question, answer)

    print("Question:", question)
    print("Answer:", answer)
    print("truth_score:", round(result["truth_score"], 4))
    print("decision:", result["decision"])
    print()

    for i, claim in enumerate(result["claims"], start=1):
        print(f"--- Claim {i} ---")
        print("text:", claim["text"][:200] + ("…" if len(claim["text"]) > 200 else ""))
        print("label:", claim["label"])
        print("confidence:", round(claim["confidence"], 4))
        print("rationale:", claim.get("rationale", ""))
        ev = claim.get("evidence") or []
        if ev:
            snippet = (ev[0].get("text") or "")[:240]
            print("top evidence:", snippet + ("…" if len(snippet) == 240 else ""))
        print()


if __name__ == "__main__":
    main()
