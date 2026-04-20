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

## Quick Start

```python
from truthscore import TruthScorer

# Initialize scorer
scorer = TruthScorer()

# Evaluate an answer
result = scorer.score(
    question="Does vitamin C prevent the common cold?",
    answer="Vitamin C prevents the common cold."
)

# Access results (0.2.0 claim-grounded result shape)
print(f"Truth Score: {result['truth_score']:.3f}")
print(f"Decision: {result['decision']}")
print(f"Claims scored: {len(result['claims'])}")
print(f"Contradictions: {result['contradictions']}")
print(f"Unsupported ratio: {result['unsupported_ratio']:.3f}")
print(f"Mean claim confidence: {result['evidence_score']:.3f}")
print(f"Coverage: {result['coverage']:.3f}")
print(f"Linguistic risk: {result['linguistic_risk']:.3f}")
print(f"Consistency score: {result['consistency_score']:.3f}")
```

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
│   └── production_example.py
├── tests/
├── paper.md, paper.bib      # JOSS-style paper sources
├── README.md
└── pyproject.toml           # Version 0.2.0
```

## Evidence grounding (replacing `DEFAULT_PASSAGES`)

The bundled `truthscore.default_corpus.DEFAULT_PASSAGES` is a **small demo seed**
so the default `TruthScorer()` has something to retrieve against. It does **not**
represent “the world,” Wikipedia, or scholarly consensus.

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

## Production mode

`create_production_scorer()` wires **real retrieval** (live **Wikipedia** via the
MediaWiki API, or a **file-backed corpus**) with a **default claim judge that does
not call any cloud LLM**: it uses ``SimilarityEvidenceVerifier`` (lexical /
structural checks over retrieved passages). No API keys are required for that
default path.

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

See `truthscore/production.py` and `examples/production_example.py`.

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

