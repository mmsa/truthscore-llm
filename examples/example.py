"""
Example usage of claim-grounded TruthScorer.
"""

import json

from truthscore import TruthScorer


def main():
    scorer = TruthScorer()

    examples = [
        (
            "Does vitamin C prevent the common cold?",
            "Vitamin C prevents the common cold.",
        ),
        (
            "What is the capital of France?",
            "The capital of France is Paris, a major European city known for its history and culture.",
        ),
        (
            "What causes climate change?",
            "Greenhouse gas emissions from human activity are the dominant driver of recent global warming.",
        ),
    ]

    for question, answer in examples:
        print("=" * 60)
        print("Q:", question)
        print("A:", answer)
        r = scorer.score(question=question, answer=answer)
        slim = {
            k: r[k]
            for k in (
                "truth_score",
                "decision",
                "unsupported_ratio",
                "contradictions",
                "consistency_score",
                "linguistic_risk",
                "emotional_intensity",
                "coverage",
                "evidence_score",
            )
        }
        print(json.dumps(slim, indent=2))
        print("claims:", [c["label"] for c in r["claims"]])


if __name__ == "__main__":
    main()
