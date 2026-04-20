# TruthScore-LLM

**Current release: 0.2.0** — Claim-grounded truth evaluation for LLM outputs: per-claim retrieval and verification, explicit risk signals (unsupported claims, contradictions, linguistic overclaim), and an optional **production** path (Wikipedia or file corpus + similarity or OpenAI judge) without requiring cloud APIs by default.

A research-oriented Python library for evaluating how well model answers are supported by evidence, when to accept or refuse, and how to wire real corpora and judges for closer-to-deployment experiments.

## Overview

From **0.2.0** onward, the default **`TruthScorer`** path is **claim-grounded**: the answer is split into claims, evidence is retrieved **per claim**, each claim is verified (supported / unsupported / contradicted), and scores aggregate with penalties for weak or conflicting evidence. On top of that, the library still exposes familiar aggregate signals:

- **Evidence / verification**: Per-claim support vs. retrieved passages (similarity-based bootstrap judge by default; optional LLM-as-judge).
- **Coverage**: How well evidence spans the answer’s claims.
- **Consistency (optional)**: If you pass a `sample_generator`, multi-sample claim consistency can be blended into the score.
- **Linguistic risk**: Hedging, overclaim, and related cues feed an explicit **linguistic risk** signal.

These signals feed a single **truth score** (0.0 to 1.0) and a categorical **decision** (ACCEPT, QUALIFIED, or REFUSE).

**What’s new in 0.2.0**

- Production wiring via **`create_production_scorer()`** with **`WikipediaRetriever`** or a **file-backed corpus**, default **`SimilarityEvidenceVerifier`** (no API key).
- Optional **`OpenAIClaimVerifier`** with `pip install 'truthscore-llm[judge]'` and `TRUTHSCORE_JUDGE=openai`.
- **`experiments/`** package (manual and API-driven runs) installable with `pip install 'truthscore-llm[experiments]'` for OpenAI-backed scripts.
- JOSS-style manuscript sources in-repo: **`paper.md`**, **`paper.bib`**.

## Where does evidence come from?

Scores are only as good as the **text the retriever returns** for each claim. The library never “knows the world” by itself—it checks the answer against **that** evidence.

| What you call | Where retrieval runs | Uses Wikipedia / network? |
|---------------|----------------------|---------------------------|
| **`TruthScorer()`** (defaults) | Bundled **`DEFAULT_PASSAGES`** in `truthscore/default_corpus.py` (small TF–IDF demo list) | **No** — fully offline |
| **`TruthScorer(retriever=...)`** | Your retriever (e.g. `TfidfPassageRetriever(passages)`, `WikipediaRetriever`, FAISS, API wrapper) | **Only if your code** uses Wikipedia or another HTTP API |
| **`create_production_scorer()`** | **Wikipedia** (MediaWiki API) when `TRUTHSCORE_EVIDENCE_MODE=wikipedia` (default), or a **local file corpus** when `mode=corpus` | **Yes** for Wikipedia mode (set `TRUTHSCORE_USER_AGENT`); **no** for file corpus |

So: **plain `TruthScorer()` does not check Wikipedia.** For live Wikipedia grounding, use **Production mode** below (or pass a `WikipediaRetriever` yourself).

Copy-paste setups for each pattern (defaults, corpus files, Wikipedia, FAISS, OpenAI judge, `CallableClaimVerifier`, `sample_generator`, env-based production) are in the **[Configuration cookbook](#configuration-cookbook-runnable-examples)**.

## Installation

### PyPI Installation

The library is available on PyPI and can be installed via:

```bash
pip install truthscore-llm
```

### Optional extras

| Extra | Purpose |
|--------|---------|
| `dev` | Tests (`pytest`, coverage) |
| `judge` | OpenAI-compatible LLM-as-judge (`openai`) |
| `retrieval` | FAISS + sentence-transformers helpers |
| `nli` | Transformers + PyTorch for entailment-style models |
| `experiments` | OpenAI for bundled `experiments/` scripts |

Examples:

```bash
pip install 'truthscore-llm[dev,judge]'
pip install 'truthscore-llm[retrieval]'
pip install 'truthscore-llm[experiments]'
```

### Development Installation

To install the library in development mode:

```bash
git clone https://github.com/mmsa/truthscore-llm.git
cd truthscore-llm
pip install -e .
```

## Quick start (default scorer — **demo corpus, not Wikipedia**)

```python
from truthscore import TruthScorer

scorer = TruthScorer()  # uses DEFAULT_PASSAGES only — see table above

# Use an answer that matches the bundled demo text (e.g. Paris, climate, water, AI).
result = scorer.score(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
)

print(f"Decision: {result['decision']}")           # often ACCEPT for this pair
print(f"Truth score: {result['truth_score']:.3f}")
print(f"Claims: {len(result['claims'])}  label={result['claims'][0]['label']}")
# Inspect per-claim evidence and verifier rationale:
# print(result["claims"][0])
```

**Same API, overstated answer on a nuanced topic** (still **only** the demo list—**not** Wikipedia):

```python
harsh = scorer.score(
    question="Does vitamin C prevent the common cold?",
    answer="Vitamin C prevents the common cold.",
)
soft = scorer.score(
    question="Does vitamin C prevent the common cold?",
    answer="Vitamin C supplementation has not been shown to prevent the common cold in the general population.",
)
# `harsh` often REFUSE / low score: unsupported vs passages + high linguistic overclaim.
# `soft` matches a sentence in DEFAULT_PASSAGES → typically much higher / ACCEPT.
```

**You supply the evidence** (retrieval is skipped; **every claim** reuses this list—typical when passages come from your DB or an API you control):

```python
result = scorer.score(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
    evidence=[{"text": "The capital of France is Paris, the largest city in metropolitan France."}],
)
# The verdict uses only these dicts (must include a "text" key). It does not call Wikipedia.
```

### Interpreting results

- **`truth_score` / `decision`** reflect **support vs. the retrieved passages**, not absolute real-world truth.
- **`TruthScorer()`** with defaults: a “bad” result usually means **your answer does not align with the tiny demo corpus** (or is penalized as linguistically overconfident), not that the package is broken.
- For **encyclopedic** grounding over many topics, use **`create_production_scorer()`** (Wikipedia) or your own **`retriever=`** / **`evidence=`**.

### Output format

`score()` returns a dictionary including per-claim audit data:

```python
{
    "truth_score": float,           # [0.0, 1.0]
    "decision": str,                # "ACCEPT" | "QUALIFIED" | "REFUSE"
    "claims": list,               # Per-claim labels, confidence, evidence (dicts)
    "unsupported_ratio": float,
    "contradictions": int,
    "consistency_score": float,
    "multi_sample_used": bool,
    "linguistic_risk": float,
    "linguistic_breakdown": dict,
    "emotional_intensity": float,
    "coverage": float,
    "evidence_score": float,        # Mean verifier confidence over claims
    "claim_raw_mean": float,
    # Back-compat aliases (deprecated but still set):
    "consistency": float,
    "language_confidence": float,
}
```

## Configuration

You can customize scoring behavior by providing a custom configuration:

```python
from truthscore import TruthScorer, TruthScoreConfig

# Create custom configuration (defaults shown; claim-level penalties are weighted here)
config = TruthScoreConfig(
    accept_threshold=0.72,
    qualified_threshold=0.52,
    top_k=5,
    claim_min_words=3,
    penalty_unsupported_ratio=0.22,
    penalty_contradiction=0.18,
    penalty_overclaim=0.14,
    consistency_blend_weight=0.08,
    claim_consistency_samples=5,
)

# Initialize scorer with custom config
scorer = TruthScorer(config=config)
```

## Project structure

```
truthscore-llm/
├── truthscore/              # Installable package
│   ├── score.py             # TruthScorer (claim-grounded)
│   ├── production.py        # create_production_scorer
│   ├── wikipedia_retriever.py
│   ├── claim_verifier.py    # Similarity / OpenAI / callable judges
│   ├── claim_extractor.py, claim_consistency.py
│   ├── retrieve.py, io_corpus.py, default_corpus.py
│   ├── config.py, types.py, coverage.py, linguistic_risk.py
│   ├── nli.py, consistency.py
│   └── ...
├── experiments/             # Research scripts (optional [experiments] extra)
├── examples/
│   ├── example.py
│   ├── production_example.py
│   └── wikipedia_example.py
├── tests/
├── paper.md, paper.bib      # JOSS-style paper sources
├── README.md
└── pyproject.toml           # Version 0.2.0
```

## Evidence grounding (replacing `DEFAULT_PASSAGES`)

The bundled `truthscore.default_corpus.DEFAULT_PASSAGES` is a **small demo seed**
so the default `TruthScorer()` has something to retrieve against. It does **not**
represent “the world,” **does not query Wikipedia**, and is not a substitute for scholarly or domain sources. See **[Where does evidence come from?](#where-does-evidence-come-from)** above.

For real use, you supply evidence from a source that matches your risk and domain:

| Source | Typical use | Caveats |
|--------|-------------|---------|
| **Wikipedia** (API or dumps) | Broad encyclopedic facts | Not authoritative for medical/legal edge cases; latency and licensing |
| **Semantic Scholar / OpenAlex / PubMed** | Papers, metadata, abstracts | Coverage varies; full text often paywalled |
| **Web search APIs** | Fresh, wide recall | Noisy snippets; ranking ≠ truth |
| **Your documents** | Policies, FAQs, internal KB | Best when “truth” is defined by your org |

Wire your corpus into a retriever (same interface as the default TF–IDF retriever):

```python
from truthscore import TruthScorer, TfidfPassageRetriever, load_passages_from_file

passages = load_passages_from_file("/path/to/passages.jsonl")  # or build list in code
scorer = TruthScorer(retriever=TfidfPassageRetriever(passages))
```

For large-scale semantic search, install optional retrieval extras and use
`build_faiss_retriever` from `truthscore.retrieve`, or implement a small class with
`retrieve(self, query: str, top_k: int) -> list[dict]` that calls any search API
and returns dicts with at least `"text"` (and ideally `"source"`, `"relevance"`).

Pair a serious corpus with a serious verifier (e.g. `OpenAIClaimVerifier` under
the optional `judge` extra), not the default similarity-only bootstrap judge.

## Production mode (Wikipedia or file corpus — **this** is the Wikipedia path)

`create_production_scorer()` wires **real retrieval**: live **Wikipedia** via the
MediaWiki API (default), or a **file-backed corpus** when you set `corpus` mode.
This is the supported way to get **network-backed** encyclopedic passages without
building a retriever by hand.

It uses a **default claim judge that does not call any cloud LLM**:
``SimilarityEvidenceVerifier`` (lexical / structural checks over retrieved passages).
No API keys are required for that default path.

```bash
pip install truthscore-llm
# Recommended when using Wikipedia:
export TRUTHSCORE_USER_AGENT="MyProduct/1.0 (https://example.com; contact@example.com)"
```

| Variable | Role |
|----------|------|
| `TRUTHSCORE_EVIDENCE_MODE` | `wikipedia` (default) or `corpus` |
| `TRUTHSCORE_CORPUS_PATH` | Required for `corpus`: `.jsonl` or `.txt` passages |
| `TRUTHSCORE_JUDGE` | `similarity` (default) or optional `openai` |
| `TRUTHSCORE_WIKIPEDIA_LANG` | Wikipedia language code (default `en`) |

Optional **OpenAI-compatible** chat judge (only if you set `TRUTHSCORE_JUDGE=openai`):

```bash
pip install 'truthscore-llm[judge]'
export OPENAI_API_KEY="sk-..."
# optional: OPENAI_BASE_URL, TRUTHSCORE_MODEL
```

```python
from truthscore import create_production_scorer, TruthScorer
from truthscore.claim_verifier import CallableClaimVerifier  # or your own class

# Default: Wikipedia + similarity judge (no OpenAI)
scorer = create_production_scorer()

# Your own verifier (any local model, HTTP API, etc.)
scorer = create_production_scorer(verifier=CallableClaimVerifier(my_fn))

out = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

See `truthscore/production.py`, `examples/production_example.py`, and `examples/wikipedia_example.py` (live Wikipedia + similarity judge; needs network).

## Configuration cookbook (runnable examples)

Each snippet is **self-contained** after `pip install truthscore-llm` unless noted. Optional extras: `[judge]`, `[retrieval]`. **Wikipedia** examples need **network** access; set a real `TRUTHSCORE_USER_AGENT` in production.

### 1. Default scorer (bundled demo corpus)

Uses `DEFAULT_PASSAGES` only — **offline**, **not Wikipedia**. See [Quick start](#quick-start-default-scorer--demo-corpus-not-wikipedia).

### 2. Custom thresholds (`TruthScoreConfig`)

```python
from truthscore import TruthScorer, TruthScoreConfig

config = TruthScoreConfig(
    accept_threshold=0.72,
    qualified_threshold=0.52,
    top_k=5,
    penalty_unsupported_ratio=0.22,
)
scorer = TruthScorer(config=config)
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

### 3. Custom TF‑IDF corpus (in-memory list)

```python
from truthscore import TruthScorer, TfidfPassageRetriever

passages = [
    "The capital of France is Paris.",
    "Berlin is the capital of Germany.",
]
scorer = TruthScorer(retriever=TfidfPassageRetriever(passages))
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

### 4. Corpus loaded from `.jsonl` or `.txt`

```python
from pathlib import Path
import tempfile
from truthscore import TruthScorer, TfidfPassageRetriever, load_passages_from_file

tmp = Path(tempfile.mkdtemp()) / "evidence.jsonl"
tmp.write_text('{"text": "The capital of France is Paris."}\n', encoding="utf-8")

passages = load_passages_from_file(tmp)
scorer = TruthScorer(retriever=TfidfPassageRetriever(passages))
result = scorer.score("France?", "Paris is the capital of France.")
```

### 5. Caller-supplied evidence (skip retrieval)

Same `TruthScorer`; every claim reuses the list you pass. Dicts must include **`"text"`**.

```python
from truthscore import TruthScorer

scorer = TruthScorer()
result = scorer.score(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
    evidence=[{"text": "The capital of France is Paris, the largest city in metropolitan France."}],
)
```

### 6. Wikipedia retriever + similarity judge (explicit, no `create_production_scorer`)

Requires **HTTP** to `*.wikipedia.org`. Set a descriptive User-Agent.

```python
from truthscore import TruthScorer
from truthscore.wikipedia_retriever import WikipediaRetriever
from truthscore.claim_verifier import SimilarityEvidenceVerifier

retriever = WikipediaRetriever(
    lang="en",
    user_agent="MyApp/1.0 (https://example.com; you@example.com)",
)
scorer = TruthScorer(
    retriever=retriever,
    verifier=SimilarityEvidenceVerifier(retriever),
)
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

### 7. FAISS + sentence-transformers retriever (optional extra)

Install: `pip install 'truthscore-llm[retrieval]'` (downloads embedding weights on first use).

`build_faiss_retriever` returns an object with **`retrieve` only**. `SimilarityEvidenceVerifier` also needs a **`similarity(a, b)`** method on the same object passed to it, so wrap retrieval with token cosine (or use an LLM verifier instead).

```python
from truthscore import TruthScorer, build_faiss_retriever
from truthscore.claim_verifier import SimilarityEvidenceVerifier
from truthscore.retrieve import pairwise_token_cosine

passages = [
    "The capital of France is Paris.",
    "Madrid is the capital of Spain.",
]

class FaissRetrieverWithCosine:
    def __init__(self, inner):
        self._inner = inner

    def retrieve(self, query: str, top_k: int = 5):
        return self._inner.retrieve(query, top_k)

    def similarity(self, a: str, b: str) -> float:
        return pairwise_token_cosine(a, b)

inner = build_faiss_retriever(passages)
retriever = FaissRetrieverWithCosine(inner)
scorer = TruthScorer(
    retriever=retriever,
    verifier=SimilarityEvidenceVerifier(retriever),
)
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

### 8. Custom judge (`CallableClaimVerifier`)

Your function: `(claim, evidence, *, question="")` → `ClaimRecord`.

```python
from truthscore import TruthScorer, TfidfPassageRetriever, CallableClaimVerifier
from truthscore.types import ClaimLabel, ClaimRecord

def my_verify(claim, evidence, *, question=""):
    # Example: conservative stub — replace with your model / API.
    return ClaimRecord(
        text=claim,
        label=ClaimLabel.UNSUPPORTED,
        confidence=0.5,
        evidence=list(evidence),
        rationale="Custom verifier placeholder.",
    )

passages = ["The capital of France is Paris."]
scorer = TruthScorer(
    retriever=TfidfPassageRetriever(passages),
    verifier=CallableClaimVerifier(my_verify),
)
result = scorer.score("France?", "Paris is the capital of France.")
```

### 9. Multi-sample claim consistency (`sample_generator`)

`sample_generator(question) -> str` is called several times; claim sets are compared for stability and blended into the score (see `TruthScoreConfig.consistency_blend_weight`).

```python
from truthscore import TruthScorer, TfidfPassageRetriever

passages = ["The capital of France is Paris."]

def cheap_stub(question: str) -> str:
    return "The capital of France is Paris."

scorer = TruthScorer(
    retriever=TfidfPassageRetriever(passages),
    sample_generator=cheap_stub,
)
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
assert result["multi_sample_used"] is True
```

### 10. Production: file-backed corpus

```python
from pathlib import Path
import tempfile
from truthscore import create_production_scorer

tmp = Path(tempfile.mkdtemp()) / "corpus.txt"
tmp.write_text(
    "The capital of France is Paris.\n\n"
    "Berlin is the capital of Germany.\n",
    encoding="utf-8",
)

scorer = create_production_scorer(evidence_mode="corpus", corpus_path=str(tmp))
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

### 11. Production: Wikipedia (explicit parameters)

Same as setting `TRUTHSCORE_EVIDENCE_MODE=wikipedia` but **without** relying on env vars for mode.

```python
from truthscore import create_production_scorer

scorer = create_production_scorer(
    evidence_mode="wikipedia",
    wikipedia_lang="en",
    wikipedia_user_agent="MyApp/1.0 (https://example.com; you@example.com)",
    judge="similarity",
)
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

### 12. Production: OpenAI LLM-as-judge (optional extra)

Requires `pip install 'truthscore-llm[judge]'` and `OPENAI_API_KEY` (or pass `openai_api_key=`). Calls the **OpenAI-compatible** chat API.

```python
import os
from truthscore import create_production_scorer

# os.environ["OPENAI_API_KEY"] = "sk-..."

scorer = create_production_scorer(
    evidence_mode="wikipedia",
    wikipedia_user_agent="MyApp/1.0 (https://example.com; you@example.com)",
    judge="openai",
    # openai_api_key="...",  # optional if env is set
)
result = scorer.score("What is the capital of France?", "The capital of France is Paris.")
```

### 13. Production + env vars (alternative to explicit kwargs)

Equivalent to many of the options above; useful in deployment.

```bash
export TRUTHSCORE_EVIDENCE_MODE=wikipedia   # or corpus
export TRUTHSCORE_USER_AGENT="MyApp/1.0 (https://example.com; you@example.com)"
# For corpus mode:
# export TRUTHSCORE_CORPUS_PATH=/path/to/passages.txt
# Optional OpenAI judge:
# export TRUTHSCORE_JUDGE=openai
# export OPENAI_API_KEY=sk-...
```

```python
from truthscore import create_production_scorer

scorer = create_production_scorer()  # reads environment
```

### Summary

| Goal | Primary API |
|------|----------------|
| Offline demo / tests | `TruthScorer()` |
| Your documents, TF‑IDF | `TruthScorer(retriever=TfidfPassageRetriever(...))` |
| Your documents, embeddings | `build_faiss_retriever` + `TruthScorer` |
| Fixed evidence you pass in | `score(..., evidence=[{"text": "..."}])` |
| Live Wikipedia + default judge | `create_production_scorer(evidence_mode="wikipedia", ...)` or `WikipediaRetriever` + `SimilarityEvidenceVerifier` |
| Live Wikipedia + LLM judge | `create_production_scorer(..., judge="openai")` + `[judge]` |
| File corpus + production wiring | `create_production_scorer(evidence_mode="corpus", corpus_path=...)` |
| Your own verifier | `TruthScorer(..., verifier=CallableClaimVerifier(fn))` or `create_production_scorer(verifier=...)` |
| Multi-sample stability | `TruthScorer(..., sample_generator=fn)` |

## Experiments (optional)

The `experiments/` directory contains helpers for reproducible comparisons (e.g. manual Q&A templates, API-driven runs). Install OpenAI support and run modules from the repo root, for example:

```bash
pip install -e ".[experiments]"
python -m experiments.setup_api
```

## Running tests

```bash
python -m pytest tests/
```

Or using unittest:

```bash
python -m unittest tests.test_score
```

## Running examples

```bash
python examples/example.py
python examples/production_example.py
# Live Wikipedia (set TRUTHSCORE_USER_AGENT for production):
PYTHONPATH=. python examples/wikipedia_example.py   # from repo; or after pip install
```

## Research disclaimer

**Important**: This library is intended for **research and experimentation**. Claim splitting, similarity judges, and bundled corpora are pragmatic defaults—not guarantees of correctness.

- **Do not rely on it as the sole basis for high-stakes decisions** without domain validation, calibration on your data, and (where needed) stronger retrieval and verification stacks (e.g. trained NLI, curated corpora, human review).
- The **production** path improves grounding (Wikipedia / your files) but does not remove the need for responsible deployment review.

The modular APIs are meant so you can swap retrievers, verifiers, and claim extractors as your evaluation matures.

## Contributing

Contributions are welcome! Please ensure that:

- Code follows the existing style and structure
- All tests pass
- New features include appropriate tests
- Documentation is updated

## License

MIT License

## Citation

If you use this library in your research, please cite (version **0.2.0**):

```bibtex
@software{mostafa2025truthscore,
  title={TruthScore-LLM: A Research Library for Evaluating Truthfulness of LLM Outputs},
  author={Mostafa, Mohamed},
  year={2026},
  url={https://github.com/mmsa/truthscore-llm},
  version={0.2.0}
}
```

A longer-form write-up for submission contexts lives in **`paper.md`** (references in **`paper.bib`**).

