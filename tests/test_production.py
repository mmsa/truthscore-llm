"""Production factory, corpus I/O, and Wikipedia retriever (mocked HTTP)."""

import os
import tempfile
import unittest
from unittest.mock import patch

from truthscore.claim_verifier import SimilarityEvidenceVerifier
from truthscore.io_corpus import load_passages_from_file
from truthscore.production import create_production_scorer
from truthscore.retrieve import TfidfPassageRetriever
from truthscore.wikipedia_retriever import WikipediaRetriever


class TestProductionValidation(unittest.TestCase):
    def test_unknown_evidence_mode(self):
        with self.assertRaises(ValueError) as ctx:
            create_production_scorer(evidence_mode="bogus")
        self.assertIn("evidence_mode", str(ctx.exception).lower())

    def test_openai_judge_requires_key(self):
        with patch.dict(os.environ, {"TRUTHSCORE_JUDGE": "openai", "OPENAI_API_KEY": ""}):
            with self.assertRaises(ValueError) as ctx:
                create_production_scorer(evidence_mode="wikipedia")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_corpus_mode_requires_path(self):
        with self.assertRaises(ValueError) as ctx:
            create_production_scorer(evidence_mode="corpus", corpus_path=None)
        self.assertIn("corpus", str(ctx.exception).lower())


class TestProductionFactory(unittest.TestCase):
    def test_wikipedia_default_uses_similarity_judge(self):
        scorer = create_production_scorer(evidence_mode="wikipedia")
        self.assertIsInstance(scorer.retriever, WikipediaRetriever)
        self.assertIsInstance(scorer.verifier, SimilarityEvidenceVerifier)

    def test_corpus_mode_builds_scorer(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(
                "Paris is the capital of France and a major European city.\n"
                "London is the capital of the United Kingdom.\n"
            )
            path = f.name
        try:
            scorer = create_production_scorer(evidence_mode="corpus", corpus_path=path)
            self.assertIsInstance(scorer.retriever, TfidfPassageRetriever)
            self.assertIsInstance(scorer.verifier, SimilarityEvidenceVerifier)
        finally:
            os.unlink(path)


class TestIoCorpus(unittest.TestCase):
    def test_jsonl(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Alpha beta gamma delta epsilon zeta."}\n')
            f.write('{"content": "Second passage here with enough words."}\n')
            path = f.name
        try:
            p = load_passages_from_file(path)
            self.assertEqual(len(p), 2)
        finally:
            os.unlink(path)


class TestWikipediaRetriever(unittest.TestCase):
    def test_retrieve_parses_api(self):
        search_payload = {
            "query": {
                "search": [
                    {"pageid": 123, "title": "Paris"},
                ]
            }
        }
        extract_payload = {
            "query": {
                "pages": {
                    "123": {
                        "pageid": 123,
                        "title": "Paris",
                        "extract": "Paris is a city in France.",
                    }
                }
            }
        }

        def fake_request(params):
            if params.get("list") == "search":
                return search_payload
            if params.get("prop") == "extracts":
                return extract_payload
            raise AssertionError(f"unexpected params {params}")

        r = WikipediaRetriever(lang="en", user_agent="UnitTest/1.0")
        with patch.object(r, "_request", side_effect=fake_request):
            docs = r.retrieve("capital of France", top_k=3)
        self.assertEqual(len(docs), 1)
        self.assertIn("Paris", docs[0]["text"])
        self.assertTrue(docs[0]["source"].startswith("wikipedia:"))

    def test_similarity_on_pair(self):
        r = WikipediaRetriever(lang="en", user_agent="Test/1.0")
        s = r.similarity("Paris France capital", "The capital of France is Paris")
        self.assertGreater(s, 0.2)


if __name__ == "__main__":
    unittest.main()
