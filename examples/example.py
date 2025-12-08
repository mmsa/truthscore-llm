"""
Example usage of the TruthScore library.

This script demonstrates how to use TruthScorer to evaluate
the truthfulness of LLM-generated answers.
"""

from truthscore import TruthScorer


def main():
    """Run example truth scoring evaluations."""
    
    # Initialize scorer
    scorer = TruthScorer()
    
    # Example 1: Medical question
    print("Example 1: Medical Question")
    print("-" * 50)
    result1 = scorer.score(
        question="Does vitamin C prevent the common cold?",
        answer="Vitamin C prevents the common cold."
    )
    print(f"Question: Does vitamin C prevent the common cold?")
    print(f"Answer: Vitamin C prevents the common cold.")
    print(f"\nTruth Score: {result1['truth_score']:.3f}")
    print(f"Decision: {result1['decision']}")
    print(f"Evidence Score: {result1['evidence_score']:.3f}")
    print(f"Consistency: {result1['consistency']:.3f}")
    print(f"Language Confidence: {result1['language_confidence']:.3f}")
    print(f"Coverage: {result1['coverage']:.3f}")
    print()
    
    # Example 2: Factual question
    print("Example 2: Factual Question")
    print("-" * 50)
    result2 = scorer.score(
        question="What is the capital of France?",
        answer="The capital of France is Paris, a major European city known for its rich history and culture."
    )
    print(f"Question: What is the capital of France?")
    print(f"Answer: The capital of France is Paris, a major European city known for its rich history and culture.")
    print(f"\nTruth Score: {result2['truth_score']:.3f}")
    print(f"Decision: {result2['decision']}")
    print(f"Evidence Score: {result2['evidence_score']:.3f}")
    print(f"Consistency: {result2['consistency']:.3f}")
    print(f"Language Confidence: {result2['language_confidence']:.3f}")
    print(f"Coverage: {result2['coverage']:.3f}")
    print()
    
    # Example 3: Uncertain answer
    print("Example 3: Uncertain Answer")
    print("-" * 50)
    result3 = scorer.score(
        question="What causes climate change?",
        answer="Maybe climate change is caused by various factors, possibly including human activities, but this is uncertain."
    )
    print(f"Question: What causes climate change?")
    print(f"Answer: Maybe climate change is caused by various factors, possibly including human activities, but this is uncertain.")
    print(f"\nTruth Score: {result3['truth_score']:.3f}")
    print(f"Decision: {result3['decision']}")
    print(f"Evidence Score: {result3['evidence_score']:.3f}")
    print(f"Consistency: {result3['consistency']:.3f}")
    print(f"Language Confidence: {result3['language_confidence']:.3f}")
    print(f"Coverage: {result3['coverage']:.3f}")
    print()


if __name__ == "__main__":
    main()

