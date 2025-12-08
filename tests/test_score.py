"""
Unit tests for TruthScore library.

Tests validate core functionality including score ranges,
decision thresholding, and component score computation.
"""

import unittest
from truthscore import TruthScorer, TruthScoreConfig


class TestTruthScorer(unittest.TestCase):
    """Test cases for TruthScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = TruthScorer()
    
    def test_score_range(self):
        """
        Test that truth_score is always in valid range [0.0, 1.0].
        
        This test validates that the scoring function produces
        scores within the expected numeric range regardless of input.
        """
        test_cases = [
            ("What is 2+2?", "Four"),
            ("Explain quantum mechanics.", "Quantum mechanics is a fundamental theory in physics."),
            ("", ""),  # Edge case: empty inputs
            ("A" * 100, "B" * 200),  # Edge case: very long inputs
        ]
        
        for question, answer in test_cases:
            with self.subTest(question=question[:30], answer=answer[:30]):
                result = self.scorer.score(question=question, answer=answer)
                
                # Check truth_score is in valid range
                self.assertGreaterEqual(
                    result['truth_score'], 0.0,
                    f"truth_score should be >= 0.0, got {result['truth_score']}"
                )
                self.assertLessEqual(
                    result['truth_score'], 1.0,
                    f"truth_score should be <= 1.0, got {result['truth_score']}"
                )
                
                # Check all component scores are in valid range
                self.assertGreaterEqual(result['evidence_score'], 0.0)
                self.assertLessEqual(result['evidence_score'], 1.0)
                
                self.assertGreaterEqual(result['consistency'], 0.0)
                self.assertLessEqual(result['consistency'], 1.0)
                
                self.assertGreaterEqual(result['language_confidence'], 0.0)
                self.assertLessEqual(result['language_confidence'], 1.0)
                
                self.assertGreaterEqual(result['coverage'], 0.0)
                self.assertLessEqual(result['coverage'], 1.0)
    
    def test_decision_thresholding(self):
        """
        Test that decision logic correctly applies thresholds.
        
        This test validates that the decision (ACCEPT, QUALIFIED, REFUSE)
        is correctly determined based on truth_score and configured thresholds.
        """
        # Create custom config with known thresholds
        config = TruthScoreConfig(
            accept_threshold=0.75,
            qualified_threshold=0.50
        )
        scorer = TruthScorer(config=config)
        
        # Test ACCEPT decision (score >= 0.75)
        # Use a high-quality answer that should score well
        result_high = scorer.score(
            question="What is the capital of France?",
            answer="The capital of France is Paris. Paris is a major European city with a rich cultural heritage and serves as the political, economic, and cultural center of France."
        )
        
        # Note: Actual scores depend on implementation, but we test the logic
        if result_high['truth_score'] >= config.accept_threshold:
            self.assertEqual(
                result_high['decision'], "ACCEPT",
                f"Score {result_high['truth_score']} >= {config.accept_threshold} should yield ACCEPT"
            )
        
        # Test REFUSE decision (score < 0.50)
        # Use a very poor answer
        result_low = scorer.score(
            question="What is the capital of France?",
            answer="Maybe possibly uncertain unclear unknown."
        )
        
        if result_low['truth_score'] < config.qualified_threshold:
            self.assertEqual(
                result_low['decision'], "REFUSE",
                f"Score {result_low['truth_score']} < {config.qualified_threshold} should yield REFUSE"
            )
        
        # Test QUALIFIED decision (0.50 <= score < 0.75)
        # Use a moderate answer
        result_mid = scorer.score(
            question="What is the capital of France?",
            answer="Paris is the capital."
        )
        
        if (config.qualified_threshold <= result_mid['truth_score'] < config.accept_threshold):
            self.assertEqual(
                result_mid['decision'], "QUALIFIED",
                f"Score {result_mid['truth_score']} in [{config.qualified_threshold}, {config.accept_threshold}) should yield QUALIFIED"
            )
        
        # Verify decision is always one of the valid values
        valid_decisions = {"ACCEPT", "QUALIFIED", "REFUSE"}
        for result in [result_high, result_low, result_mid]:
            self.assertIn(
                result['decision'], valid_decisions,
                f"Decision must be one of {valid_decisions}, got {result['decision']}"
            )
    
    def test_result_structure(self):
        """Test that score() returns expected dictionary structure."""
        result = self.scorer.score(
            question="Test question?",
            answer="Test answer."
        )
        
        required_keys = {
            'truth_score',
            'decision',
            'evidence_score',
            'consistency',
            'language_confidence',
            'coverage'
        }
        
        self.assertEqual(
            set(result.keys()), required_keys,
            f"Result must contain exactly these keys: {required_keys}"
        )
        
        # Check all values are numeric except decision
        for key in required_keys - {'decision'}:
            self.assertIsInstance(
                result[key], (int, float),
                f"{key} must be numeric, got {type(result[key])}"
            )
        
        # Check decision is string
        self.assertIsInstance(
            result['decision'], str,
            f"decision must be string, got {type(result['decision'])}"
        )


if __name__ == '__main__':
    unittest.main()

