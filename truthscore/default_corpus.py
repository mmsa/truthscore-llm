"""
Tiny **demo** passages bundled with the package so ``TruthScorer()`` works out of
the box for tests and examples.

**This cannot (and must not) try to “cover everything in the world”.** No finite
``DEFAULT_PASSAGES`` list is a substitute for a real evidence layer. Truthfulness
is always relative to *what you retrieve*.

Reasonable production options (you implement or plug in as ``retriever=``):

- **Wikipedia (or other encyclopedia)**: good for general encyclopedic facts;
  use the REST API, downloadable dumps, or a hosted snapshot; respect licenses
  and rate limits; expect gaps and occasional errors.
- **Academic corpora**: **Semantic Scholar**, **OpenAlex**, **PubMed**, or
  **Google Scholar** (often via third-party APIs or careful scraping policies)—
  better for citations and biomedical claims; not a full world model.
- **Web search**: Bing / Google / Brave / etc. APIs return snippets you index or
  pass through; noisy and time-sensitive, but broad coverage.
- **Your own KB**: policies, product docs, internal wikis—usually the best
  source for domain-specific “truth”.

Pattern: build a list (or index) of passages from your source, then::

    from truthscore import TruthScorer, TfidfPassageRetriever
    passages = [...]  # from DB, API, dump, etc.
    scorer = TruthScorer(retriever=TfidfPassageRetriever(passages))

For large corpora, use ``build_faiss_retriever`` (see ``retrieve.py``) with the
optional ``[retrieval]`` extras, or any object with a ``retrieve(query, top_k)``
method returning ``{"text", "source", "relevance"}`` dicts.
"""

DEFAULT_PASSAGES: list = [
    "The capital of France is Paris, the largest city in metropolitan France.",
    "Paris is known for European culture, major historical landmarks, and rich artistic heritage.",
    "France is a country in Western Europe; its government seat is in Paris.",
    "Photosynthesis is the process used by plants to convert light into chemical energy.",
    "Photosynthesis occurs in chloroplasts and produces glucose and oxygen from CO2 and water.",
    "Vitamin C supplementation has not been shown to prevent the common cold in the general population.",
    "Climate change is driven largely by greenhouse gas emissions from human activities.",
    "Recent global warming is strongly linked to greenhouse gas emissions from human activity.",
    "Carbon dioxide and methane are important greenhouse gases that trap heat.",
    "Water is a liquid at room temperature and is essential for life on Earth.",
    "Artificial intelligence is a branch of computer science concerned with intelligent behavior in machines.",
]
