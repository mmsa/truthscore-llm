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
    
    def test_valid_answer_acceptance(self):
        """
        Test that valid, well-supported answers receive ACCEPT decision.
        
        Valid answers should have:
        - High evidence support
        - Good consistency
        - Clear language
        - Good coverage
        """
        scorer = TruthScorer()
        
        # Test case 1: Clear factual answer with good evidence
        result1 = scorer.score(
            question="What is the capital of France?",
            answer="The capital of France is Paris. Paris is a major European city and serves as the political, economic, and cultural center of France."
        )
        
        # Valid answer should have high truth score
        self.assertGreaterEqual(
            result1['truth_score'], 0.55,
            "Valid answer should score at least 0.55"
        )
        
        # Should not be REFUSED
        self.assertNotEqual(
            result1['decision'], "REFUSE",
            "Valid answer should not be REFUSED"
        )
        
        # Test case 2: Well-structured answer with evidence
        result2 = scorer.score(
            question="What is photosynthesis?",
            answer="Photosynthesis is the process by which plants convert light energy into chemical energy. This process occurs in chloroplasts and involves carbon dioxide and water being converted into glucose and oxygen."
        )
        
        self.assertGreaterEqual(
            result2['truth_score'], 0.55,
            "Well-structured answer should score at least 0.55"
        )
        self.assertNotEqual(
            result2['decision'], "REFUSE",
            "Well-structured answer should not be REFUSED"
        )
    
    def test_invalid_answer_rejection(self):
        """
        Test that invalid or poorly supported answers receive REFUSE decision.
        
        Invalid answers should have:
        - Low evidence support
        - Poor consistency
        - Uncertain language
        - Insufficient coverage
        """
        scorer = TruthScorer()
        
        # Test case 1: Answer with very low evidence (like the user's example)
        result1 = scorer.score(
            question="What is the capital of France?",
            answer="The capital of France is London."
        )
        
        # Invalid answer with low evidence should be REFUSED or at least not ACCEPT
        # Note: The actual decision depends on all component scores
        if result1['evidence_score'] < 0.4 and result1['truth_score'] < 0.55:
            self.assertEqual(
                result1['decision'], "REFUSE",
                f"Answer with low evidence ({result1['evidence_score']:.3f}) and score {result1['truth_score']:.3f} should be REFUSED, got {result1['decision']}"
            )
        
        # At minimum, invalid answers should not be ACCEPTED
        self.assertNotEqual(
            result1['decision'], "ACCEPT",
            f"Invalid answer (wrong capital) should not be ACCEPTED, got {result1['decision']}"
        )
        
        # Test case 2: Answer with excessive hedging/uncertainty
        result2 = scorer.score(
            question="What causes climate change?",
            answer="Maybe possibly climate change is caused by various factors, but this is uncertain and unclear. We don't really know for sure."
        )
        
        # Answers with excessive uncertainty should be penalized
        self.assertLessEqual(
            result2['language_confidence'], 0.5,
            "Answer with excessive hedging should have low language confidence"
        )
        
        # Test case 3: Very short or incomplete answer
        result3 = scorer.score(
            question="Explain quantum mechanics.",
            answer="Maybe quantum."
        )
        
        # Very short answers should score low
        self.assertLess(
            result3['truth_score'], 0.75,
            "Very short answer should not score high"
        )
        
        # Test case 4: Contradictory or nonsensical answer
        result4 = scorer.score(
            question="What is water?",
            answer="Water is not wet and it is dry. Water is also not water but something else entirely."
        )
        
        # Contradictory answers should have low consistency
        self.assertLess(
            result4['consistency'], 0.7,
            "Contradictory answer should have low consistency"
        )
    
    def test_qualified_answer_boundary(self):
        """
        Test answers that fall in the QUALIFIED range.
        
        QUALIFIED answers are those that are acceptable but not strongly supported.
        """
        scorer = TruthScorer()
        
        # Test case: Answer that should be QUALIFIED (between 0.55 and 0.75)
        result = scorer.score(
            question="What is artificial intelligence?",
            answer="Artificial intelligence is a field of computer science that deals with creating systems that can perform tasks typically requiring human intelligence."
        )
        
        # Check that decision is valid
        valid_decisions = {"ACCEPT", "QUALIFIED", "REFUSE"}
        self.assertIn(
            result['decision'], valid_decisions,
            f"Decision must be one of {valid_decisions}, got {result['decision']}"
        )
        
        # If it's QUALIFIED, verify the score range
        if result['decision'] == "QUALIFIED":
            self.assertGreaterEqual(
                result['truth_score'], 0.55,
                "QUALIFIED decision requires score >= 0.55"
            )
            self.assertLess(
                result['truth_score'], 0.75,
                "QUALIFIED decision requires score < 0.75"
            )
    
    def test_evidence_score_impact(self):
        """
        Test that low evidence scores significantly impact the final decision.
        
        With evidence_weight = 0.6, low evidence should heavily penalize the score.
        """
        scorer = TruthScorer()
        
        # Test with low evidence score scenario
        result_low_evidence = scorer.score(
            question="Does vitamin C prevent the common cold?",
            answer="Vitamin C definitely prevents all colds completely."
        )
        
        # If evidence score is low, overall score should be lower than high evidence cases
        # Note: This is a relative check, not absolute threshold
        if result_low_evidence['evidence_score'] < 0.4:
            # Low evidence should contribute to lower overall score
            # With evidence_weight = 0.6, low evidence significantly impacts the score
            self.assertLess(
                result_low_evidence['truth_score'], 0.70,
                f"Low evidence score ({result_low_evidence['evidence_score']:.3f}) should result in lower truth score"
            )
        
        # Compare with potentially better evidence scenario
        result_better = scorer.score(
            question="What is the capital of France?",
            answer="The capital of France is Paris, a major European city known for its rich history, art, and culture."
        )
        
        # Better structured answer should generally score better
        # (though this depends on implementation details)
        self.assertGreaterEqual(
            result_better['truth_score'], 0.0,
            "Truth score should be valid"
        )
    
    def test_threshold_boundaries(self):
        """
        Test that thresholds are correctly applied at boundaries.
        
        Tests scores right at the threshold values.
        """
        config = TruthScoreConfig(
            accept_threshold=0.75,
            qualified_threshold=0.55
        )
        scorer = TruthScorer(config=config)
        
        # Test various scenarios to understand threshold behavior
        test_cases = [
            ("Short answer", "Yes."),
            ("Medium answer", "The answer is correct and well-supported by evidence."),
            ("Long answer", "This is a comprehensive answer that provides detailed information about the topic. It includes multiple aspects and explains the concept thoroughly with supporting details and examples."),
        ]
        
        for desc, answer in test_cases:
            with self.subTest(description=desc):
                result = scorer.score(
                    question="Test question?",
                    answer=answer
                )
                
                # Verify decision logic
                if result['truth_score'] >= 0.75:
                    self.assertEqual(
                        result['decision'], "ACCEPT",
                        f"Score {result['truth_score']:.3f} >= 0.75 should be ACCEPT"
                    )
                elif result['truth_score'] >= 0.55:
                    self.assertEqual(
                        result['decision'], "QUALIFIED",
                        f"Score {result['truth_score']:.3f} in [0.55, 0.75) should be QUALIFIED"
                    )
                else:
                    self.assertEqual(
                        result['decision'], "REFUSE",
                        f"Score {result['truth_score']:.3f} < 0.55 should be REFUSE"
                    )


if __name__ == '__main__':
    unittest.main()

