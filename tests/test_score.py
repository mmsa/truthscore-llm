"""
Core API tests for claim-grounded TruthScorer.
"""

import unittest

from truthscore import TruthScorer, TruthScoreConfig


class TestTruthScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = TruthScorer()

    def test_score_range(self):
        cases = [
            ("What is 2+2?", "Four is the answer to two plus two."),
            ("What is the capital of France?", "The capital of France is Paris."),
            ("", ""),
        ]
        for q, a in cases:
            with self.subTest(q=q[:20], a=a[:20]):
                r = self.scorer.score(question=q, answer=a)
                self.assertGreaterEqual(r["truth_score"], 0.0)
                self.assertLessEqual(r["truth_score"], 1.0)
                self.assertIn(r["decision"], {"ACCEPT", "QUALIFIED", "REFUSE"})
                for k in (
                    "evidence_score",
                    "coverage",
                    "consistency_score",
                    "linguistic_risk",
                    "unsupported_ratio",
                ):
                    self.assertIn(k, r)
                    self.assertIsInstance(r[k], (int, float))

    def test_result_structure(self):
        r = self.scorer.score(question="Test question?", answer="Test answer with enough words.")
        required = {
            "truth_score",
            "decision",
            "claims",
            "unsupported_ratio",
            "contradictions",
            "consistency_score",
            "multi_sample_used",
            "linguistic_risk",
            "emotional_intensity",
            "coverage",
            "evidence_score",
            "claim_raw_mean",
            "consistency",
            "language_confidence",
            "linguistic_breakdown",
        }
        self.assertEqual(set(r.keys()), required)
        self.assertIsInstance(r["claims"], list)

    def test_decision_thresholds(self):
        cfg = TruthScoreConfig(accept_threshold=0.8, qualified_threshold=0.45)
        scorer = TruthScorer(config=cfg)
        r = scorer.score(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        if r["truth_score"] >= cfg.accept_threshold:
            self.assertEqual(r["decision"], "ACCEPT")
        elif r["truth_score"] >= cfg.qualified_threshold:
            self.assertEqual(r["decision"], "QUALIFIED")
        else:
            self.assertEqual(r["decision"], "REFUSE")

    def test_wrong_capital_not_accepted_default_pipeline(self):
        r = self.scorer.score(
            "What is the capital of France?",
            "The capital of France is London.",
        )
        self.assertNotEqual(r["decision"], "ACCEPT")


if __name__ == "__main__":
    unittest.main()
