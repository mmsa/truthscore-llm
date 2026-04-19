"""
Production scorer: Wikipedia (or file corpus) + default similarity judge.

No API keys required. Set TRUTHSCORE_USER_AGENT when hitting Wikipedia.

Run from repo root: PYTHONPATH=. python examples/production_example.py
"""

import os

from truthscore import create_production_scorer


def main() -> None:
    mode = os.environ.get("TRUTHSCORE_EVIDENCE_MODE", "wikipedia")
    judge = os.environ.get("TRUTHSCORE_JUDGE", "similarity")
    print(f"Evidence: {mode}, judge: {judge}")

    scorer = create_production_scorer()
    r = scorer.score(
        "What is the capital of France?",
        "The capital of France is Paris.",
    )
    print("truth_score:", r["truth_score"])
    print("decision:", r["decision"])
    print("claims:", [(c["text"][:60], c["label"]) for c in r["claims"]])


if __name__ == "__main__":
    main()
