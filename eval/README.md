# Claim-grounded evaluation (100 questions)

This directory holds a **reproducible** evaluation harness for the paper-style
metrics (overconfident errors, refusals, claim-level diagnostics).

## Files

| File | Purpose |
|------|---------|
| `data/questions_100.json` | 100 items, 20 per category (`true_fact`, `false_claim`, `mixed`, `unanswerable`, `subjective`) |
| `generate_dataset.py` | Regenerate `questions_100.json` if you edit templates |
| `run_claim_evaluation.py` | Run configs A–D and emit the aggregate JSON + JSONL raw log |

## Security

- **Never** commit API keys or paste them into source files.
- Use environment variables: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`.
- `eval/results/` is gitignored.

## Configurations

| Key | Description |
|-----|-------------|
| **A** | Vanilla answer only (no TruthScore); labels describe the raw answer. |
| **B** | `TruthScorer()` — bundled demo corpus + default similarity verifier. |
| **C** | `create_production_scorer(evidence_mode="wikipedia", …)` + similarity judge. |
| **D** | Same as **C** with `judge="openai"` (requires `[judge]` + API key). |

The **same** vanilla answer is scored under B/C/D so differences isolate **evidence + judge**, not sampling noise in generation.

## Labeling

- **Human (recommended for publication):** run without `--auto-label-with-openai`, then fill labels from your protocol and merge (script prints a warning).
- **Automated rubric (exploratory only):** `--auto-label-with-openai` calls the same API with temperature 0 and a fixed JSON rubric. This is **not** IRB human coding; disclose in the paper.

## Commands

```bash
cd /path/to/truthscore-llm
export OPENAI_API_KEY="..."   # never commit
export TRUTHSCORE_USER_AGENT="YourPaperEval/1.0 (https://...; email@...)"
pip install -e ".[judge]"

# Smoke test (no API, no Wikipedia)
PYTHONPATH=. python eval/run_claim_evaluation.py --dry-run --limit 3 --out eval/results/smoke.json

# Full run with automated labels (costs API calls; Wikipedia network)
PYTHONPATH=. python eval/run_claim_evaluation.py \
  --out eval/results/latest.json \
  --auto-label-with-openai
```

Output JSON matches the structure expected by the paper pipeline (`summary`,
`confusion_matrix`, `score_distribution`, `category_breakdown`, `sample_outputs`)
plus `raw_jsonl` path to per-question logs.

## Versioning

Pin `truthscore-llm==…` in the paper when citing numeric results.
