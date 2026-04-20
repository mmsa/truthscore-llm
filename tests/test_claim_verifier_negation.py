"""Regression tests for SimilarityEvidenceVerifier negation heuristics."""

import unittest

from truthscore.claim_verifier import SimilarityEvidenceVerifier
from truthscore.retrieve import TfidfPassageRetriever


class TestNegationHeuristics(unittest.TestCase):
    def test_annotation_substring_does_not_trigger_negation_veto(self) -> None:
        """'not' inside 'annotation' must not fire the old substring false positive."""
        passages = [
            "Mars is the fourth planet from the Sun. It is also known as the Red Planet.",
            "Annotation pipelines are complex.",
        ]
        r = TfidfPassageRetriever(passages)
        v = SimilarityEvidenceVerifier(r)
        evidence = r.retrieve("Mars is the fourth planet from the Sun", 5)
        rec = v.verify("Mars is the fourth planet from the Sun", evidence, question="")
        self.assertEqual(rec.label.value, "SUPPORTED")

    def test_negation_in_same_sentence_still_vetoes(self) -> None:
        passages = [
            "Mars is not the fourth planet from the Sun.",
        ]
        r = TfidfPassageRetriever(passages)
        v = SimilarityEvidenceVerifier(r)
        evidence = r.retrieve("Mars is the fourth planet from the Sun", 5)
        rec = v.verify("Mars is the fourth planet from the Sun", evidence, question="")
        self.assertEqual(rec.label.value, "UNSUPPORTED")
        self.assertIn("negation", rec.rationale.lower())

    def test_cannot_in_unrelated_sentence_ignored_when_best_sentence_clear(self) -> None:
        """Negation in a later sentence should not veto if the best-matching sentence supports."""
        long = (
            "Mars is the fourth planet from the Sun. "
            "Some fringe theories cannot be reconciled with observations."
        )
        r = TfidfPassageRetriever([long])
        v = SimilarityEvidenceVerifier(r)
        evidence = r.retrieve("Mars is the fourth planet from the Sun", 3)
        rec = v.verify("Mars is the fourth planet from the Sun", evidence, question="")
        self.assertEqual(rec.label.value, "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
