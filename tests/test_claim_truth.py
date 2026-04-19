"""
Claim-grounded truth scoring scenarios (deterministic verifiers / corpora).
"""

import unittest

from truthscore import TruthScorer, TruthScoreConfig
from truthscore.claim_verifier import CallableClaimVerifier
from truthscore.default_corpus import DEFAULT_PASSAGES
from truthscore.retrieve import TfidfPassageRetriever
from truthscore.types import ClaimLabel, ClaimRecord


def _record(claim: str, label: ClaimLabel, conf: float = 0.9, ev=None) -> ClaimRecord:
    return ClaimRecord(
        text=claim,
        label=label,
        confidence=conf,
        evidence=ev or [],
        rationale="test",
    )


class TestClaimTruthScenarios(unittest.TestCase):
    def test_true_factual_answer_high_score(self):
        """Well-supported factual claims → high truth_score."""
        corpus = [
            "The capital of France is Paris.",
            "Paris is the largest city in France.",
        ]
        retriever = TfidfPassageRetriever(corpus)

        def verify(claim, evidence, question=""):
            blob = " ".join(d.get("text", "") for d in evidence).lower()
            cl = claim.lower()
            if "paris" in cl and "capital" in cl and "france" in cl:
                if "paris" in blob and "capital" in blob:
                    return _record(claim, ClaimLabel.SUPPORTED, 0.92, list(evidence))
            if "paris" in cl and "largest" in cl and "france" in cl:
                if "paris" in blob and "largest" in blob:
                    return _record(claim, ClaimLabel.SUPPORTED, 0.9, list(evidence))
            return _record(claim, ClaimLabel.UNSUPPORTED, 0.2, list(evidence))

        scorer = TruthScorer(retriever=retriever, verifier=CallableClaimVerifier(verify))
        r = scorer.score(
            "What is the capital of France?",
            "The capital of France is Paris. Paris is the largest city in France.",
        )
        self.assertGreaterEqual(r["truth_score"], 0.78)
        self.assertEqual(r["contradictions"], 0)
        self.assertIn("claims", r)
        self.assertTrue(all(c["label"] == "SUPPORTED" for c in r["claims"]))

    def test_confident_wrong_answer_low_score(self):
        """Confident but false claim → low score (contradicted or unsupported)."""
        corpus = ["The capital of France is Paris."]
        retriever = TfidfPassageRetriever(corpus)

        def verify(claim, evidence, question=""):
            if "london" in claim.lower() and "france" in claim.lower():
                return _record(claim, ClaimLabel.CONTRADICTED, 0.88, list(evidence))
            return _record(claim, ClaimLabel.UNSUPPORTED, 0.2, list(evidence))

        scorer = TruthScorer(retriever=retriever, verifier=CallableClaimVerifier(verify))
        r = scorer.score(
            "What is the capital of France?",
            "The capital of France is London.",
        )
        self.assertLess(r["truth_score"], 0.45)
        self.assertGreaterEqual(r["contradictions"], 1)

    def test_mixed_answer_mid_score(self):
        """One supported and one unsupported claim → mid-range score."""
        corpus = ["Paris is the capital of France.", "Water boils at 100C at sea level."]
        retriever = TfidfPassageRetriever(corpus)

        def verify(claim, evidence, question=""):
            cl = claim.lower()
            if "mars" in cl:
                return _record(claim, ClaimLabel.UNSUPPORTED, 0.25, list(evidence))
            if "paris" in cl and "capital" in cl and "france" in cl:
                return _record(claim, ClaimLabel.SUPPORTED, 0.9, list(evidence))
            return _record(claim, ClaimLabel.UNSUPPORTED, 0.3, list(evidence))

        scorer = TruthScorer(retriever=retriever, verifier=CallableClaimVerifier(verify))
        r = scorer.score(
            "Geography?",
            "Paris is the capital of France. The capital of Mars is Paris.",
        )
        self.assertGreater(r["truth_score"], 0.25)
        self.assertLess(r["truth_score"], 0.82)
        self.assertGreater(r["unsupported_ratio"], 0.0)

    def test_no_evidence_penalized(self):
        """Empty retrieval → unsupported claims and low truth_score."""

        class EmptyRetriever:
            def retrieve(self, query: str, top_k: int = 5):
                return []

        def verify(claim, evidence, question=""):
            self.assertEqual(evidence, [])
            return _record(claim, ClaimLabel.UNSUPPORTED, 0.1, [])

        scorer = TruthScorer(
            retriever=EmptyRetriever(),
            verifier=CallableClaimVerifier(verify),
        )
        r = scorer.score(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        self.assertLess(r["truth_score"], 0.5)
        self.assertEqual(r["unsupported_ratio"], 1.0)
        self.assertLessEqual(r["coverage"], 0.05)

    def test_emotional_persuasive_nonsense_risk(self):
        """High emotional intensity + weak support raises linguistic_risk."""
        corpus = ["Water is a liquid."]
        retriever = TfidfPassageRetriever(corpus)

        def verify(claim, evidence, question=""):
            return _record(claim, ClaimLabel.UNSUPPORTED, 0.2, list(evidence))

        scorer = TruthScorer(retriever=retriever, verifier=CallableClaimVerifier(verify))
        r = scorer.score(
            "What is water?",
            "Water is ABSOLUTELY the most INCREDIBLE chemical EVER discovered!!! "
            "Everyone MUST believe this!!!",
        )
        self.assertGreater(r["emotional_intensity"], 0.35)
        self.assertGreater(r["linguistic_risk"], 0.35)


class TestMultiSampleConsistency(unittest.TestCase):
    def test_sample_generator_increases_multi_sample_flag(self):
        seq = [
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "France's capital city is Paris.",
            "Paris serves as the capital of France.",
            "It is Paris.",
        ]
        i = {"n": 0}

        def gen(q: str) -> str:
            s = seq[i["n"] % len(seq)]
            i["n"] += 1
            return s

        scorer = TruthScorer(
            retriever=TfidfPassageRetriever(DEFAULT_PASSAGES),
            sample_generator=gen,
        )
        r = scorer.score("Capital of France?", "Paris is the capital of France.")
        self.assertTrue(r["multi_sample_used"])
        self.assertGreaterEqual(r["consistency_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
