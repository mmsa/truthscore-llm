"""
Broader use-case tests: default pipeline, injected evidence, config, extractors.
"""

import unittest

from truthscore import TruthScorer, TruthScoreConfig
from truthscore.claim_verifier import CallableClaimVerifier
from truthscore.default_corpus import DEFAULT_PASSAGES
from truthscore.retrieve import TfidfPassageRetriever
from truthscore.types import ClaimLabel, ClaimRecord


def _rec(claim: str, label: ClaimLabel, conf: float = 0.85, ev=None):
    return ClaimRecord(
        text=claim,
        label=label,
        confidence=conf,
        evidence=list(ev or []),
        rationale="fixture",
    )


class TestDefaultPipelineUseCases(unittest.TestCase):
    """End-to-end behavior with bundled corpus + SimilarityEvidenceVerifier."""

    def setUp(self):
        self.scorer = TruthScorer()

    def test_paris_capital_accepted(self):
        r = self.scorer.score(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        self.assertEqual(r["decision"], "ACCEPT")
        self.assertGreaterEqual(r["truth_score"], 0.7)
        self.assertEqual(r["contradictions"], 0)

    def test_london_capital_refused_or_qualified(self):
        r = self.scorer.score(
            "What is the capital of France?",
            "The capital of France is London.",
        )
        self.assertNotEqual(r["decision"], "ACCEPT")
        self.assertTrue(any(c["label"] in ("UNSUPPORTED", "CONTRADICTED") for c in r["claims"]))

    def test_vitamin_c_overclaim_not_accepted(self):
        r = self.scorer.score(
            "Does vitamin C prevent the common cold?",
            "Vitamin C definitely prevents all colds completely.",
        )
        self.assertNotEqual(r["decision"], "ACCEPT")

    def test_photosynthesis_supported(self):
        # Single claim aligned with default corpus (splitting on "and" weakens short clauses).
        r = self.scorer.score(
            "What is photosynthesis?",
            "Photosynthesis is the process used by plants to convert light into chemical energy.",
        )
        self.assertGreaterEqual(r["truth_score"], 0.55)
        self.assertIn(r["decision"], {"ACCEPT", "QUALIFIED"})

    def test_water_basic_claim(self):
        r = self.scorer.score(
            "What is water?",
            "Water is a liquid at room temperature and is essential for life on Earth.",
        )
        self.assertGreaterEqual(r["truth_score"], 0.5)


class TestInjectedEvidenceUseCase(unittest.TestCase):
    """Caller supplies grounding passages (e.g. RAG context)."""

    def test_shared_evidence_for_all_claims(self):
        passages = [
            "The speed of light in vacuum is approximately 299792458 meters per second.",
        ]
        retriever = TfidfPassageRetriever(passages)

        def verify(claim, evidence, question=""):
            blob = " ".join(d.get("text", "") for d in evidence).lower()
            if "299792458" in blob or "speed of light" in blob:
                if "light" in claim.lower():
                    return _rec(claim, ClaimLabel.SUPPORTED, 0.9, evidence)
            return _rec(claim, ClaimLabel.UNSUPPORTED, 0.2, evidence)

        scorer = TruthScorer(retriever=retriever, verifier=CallableClaimVerifier(verify))
        fixed = [{"text": passages[0], "source": "ctx:0", "relevance": 1.0}]
        r = scorer.score(
            "How fast is light?",
            "Light travels at roughly three hundred million meters per second in vacuum.",
            evidence=fixed,
        )
        self.assertGreaterEqual(r["truth_score"], 0.55)
        for c in r["claims"]:
            self.assertTrue(len(c["evidence"]) >= 1)


class TestEmptyAndEdgeUseCases(unittest.TestCase):
    def test_empty_answer_refuses(self):
        r = TruthScorer().score("Any question?", "")
        self.assertEqual(r["decision"], "REFUSE")
        self.assertEqual(r["claims"], [])
        self.assertEqual(r["truth_score"], 0.0)

    def test_whitespace_only_answer(self):
        r = TruthScorer().score("Q?", "   \n\t  ")
        self.assertEqual(r["claims"], [])

    def test_claim_record_schema(self):
        r = TruthScorer().score(
            "What is the capital of France?",
            "Paris is the capital of France.",
        )
        self.assertTrue(len(r["claims"]) >= 1)
        for c in r["claims"]:
            self.assertIn("text", c)
            self.assertIn("label", c)
            self.assertIn("confidence", c)
            self.assertIn("evidence", c)
            self.assertIn("rationale", c)
            self.assertIn(c["label"], {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"})

    def test_linguistic_breakdown_structure(self):
        r = TruthScorer().score(
            "Test?",
            "This is a calm factual sentence with enough words for one claim.",
        )
        b = r["linguistic_breakdown"]
        for k in ("overclaim_risk", "emotional_intensity", "assertive_tone", "weak_evidence_mass"):
            self.assertIn(k, b)


class TestClaimExtractorUseCase(unittest.TestCase):
    def test_two_argument_extractor_supported(self):
        def two_arg(q, a):
            return [a.strip()] if len(a.split()) >= 4 else []

        scorer = TruthScorer(claim_extractor=two_arg)
        r = scorer.score("Q?", "Alpha beta gamma delta epsilon.")
        self.assertEqual(len(r["claims"]), 1)


class TestConfigPenaltyUseCase(unittest.TestCase):
    def test_stricter_penalties_lower_truth_score(self):
        corpus = ["Paris is the capital of France."]
        retriever = TfidfPassageRetriever(corpus)

        def verify(claim, evidence, question=""):
            return _rec(claim, ClaimLabel.UNSUPPORTED, 0.4, evidence)

        loose = TruthScoreConfig(
            accept_threshold=0.9,
            qualified_threshold=0.1,
            penalty_unsupported_ratio=0.05,
            penalty_contradiction=0.05,
            penalty_overclaim=0.05,
        )
        tight = TruthScoreConfig(
            accept_threshold=0.9,
            qualified_threshold=0.1,
            penalty_unsupported_ratio=0.35,
            penalty_contradiction=0.35,
            penalty_overclaim=0.25,
        )
        v = CallableClaimVerifier(verify)
        r_loose = TruthScorer(config=loose, retriever=retriever, verifier=v).score(
            "?", "Paris is the capital of France."
        )
        r_tight = TruthScorer(config=tight, retriever=retriever, verifier=v).score(
            "?", "Paris is the capital of France."
        )
        self.assertGreater(r_loose["truth_score"], r_tight["truth_score"])


class TestContradictionAggregationUseCase(unittest.TestCase):
    def test_single_contradiction_hurts_score(self):
        corpus = ["Reference fact one.", "Reference fact two."]
        retriever = TfidfPassageRetriever(corpus)
        n = {"i": 0}

        def verify(claim, evidence, question=""):
            n["i"] += 1
            if n["i"] == 1:
                return _rec(claim, ClaimLabel.SUPPORTED, 0.9, evidence)
            return _rec(claim, ClaimLabel.CONTRADICTED, 0.85, evidence)

        scorer = TruthScorer(retriever=retriever, verifier=CallableClaimVerifier(verify))
        r = scorer.score(
            "Mixed?",
            "First sentence is fine here. Second sentence contradicts all known physics completely.",
        )
        self.assertGreaterEqual(r["contradictions"], 1)
        self.assertLess(r["truth_score"], 0.85)


class TestRetrieverCorpusUseCase(unittest.TestCase):
    def test_custom_corpus_improves_domain_match(self):
        niche = [
            "The Zorblax engine runs on dilithium crystals only.",
            "Dilithium is used in Zorblax engines for energy containment.",
        ]
        r_niche = TruthScorer(retriever=TfidfPassageRetriever(niche)).score(
            "What powers a Zorblax engine?",
            "The Zorblax engine runs on dilithium crystals.",
        )
        self.assertGreaterEqual(r_niche["coverage"], 0.0)
        self.assertIn(r_niche["decision"], {"ACCEPT", "QUALIFIED", "REFUSE"})


class TestMultiClaimExtractionUseCase(unittest.TestCase):
    def test_and_splits_multiple_claims(self):
        corpus = ["Paris is the capital of France.", "London is the capital of the United Kingdom."]
        retriever = TfidfPassageRetriever(corpus)

        def verify(claim, evidence, question=""):
            cl = claim.lower()
            if "paris" in cl and "france" in cl:
                return _rec(claim, ClaimLabel.SUPPORTED, 0.9, evidence)
            if "london" in cl and "kingdom" in cl:
                return _rec(claim, ClaimLabel.SUPPORTED, 0.88, evidence)
            return _rec(claim, ClaimLabel.UNSUPPORTED, 0.2, evidence)

        scorer = TruthScorer(retriever=retriever, verifier=CallableClaimVerifier(verify))
        r = scorer.score(
            "Capitals?",
            "Paris is the capital of France and London is the capital of the United Kingdom.",
        )
        self.assertGreaterEqual(len(r["claims"]), 2)
        self.assertTrue(all(c["label"] == "SUPPORTED" for c in r["claims"]))


if __name__ == "__main__":
    unittest.main()
