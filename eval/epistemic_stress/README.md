# Epistemic stress evaluation (120 prompts)

Second evaluation track: **stress prompts** where models tend to overcommit (misconceptions,
false premises, unanswerable items, mixed-truth tasks, weak/conflicting evidence).

## What runs

| System | Behavior |
|--------|----------|
| `vanilla` | Single LLM answer (temperature 0.2). |
| `rag` | Top Wikipedia passages injected into the prompt, then LLM answer. |
| `self_consistency` | 5 higher-temperature samples; **majority** answer string. |
| `truthscore_similarity` | **Same** vanilla candidate answer; **gated** with `create_production_scorer(..., judge="similarity")` (Wikipedia retrieval). If `REFUSE`, a fixed refusal string is shown. |
| `truthscore_llm_judge` | Same gating with **OpenAI** claim judge (optional; needs API + `[judge]`). |

Gating matches the paper pattern: **generate once (vanilla), then score for TruthScore columns**.

## Dataset

- `data/prompts_120.json`: 6 × 20 rows with `id`, `category`, `prompt`, `expected_behavior`.
- Regenerate: `python eval/epistemic_stress/generate_dataset_120.py`

## Labels (six-way)

Primary labels are **not** human by default. Use `--auto-label-with-openai` for an
automated rubric panel (disclose in publications), or label offline and extend the runner.

## Claim-level pass

`--claim-annotate-n 40` runs an extra **Wikipedia + OpenAI judge** `score()` on each
system’s **displayed** answer for the first N prompts (costly). Omit for quick runs.

## Commands

```bash
export OPENAI_API_KEY="..."
export TRUTHSCORE_USER_AGENT="YourEval/1.0 (https://...; you@...)"
pip install -e ".[judge]"

PYTHONPATH=. python eval/epistemic_stress/run_epistemic_eval.py --dry-run --limit 4

PYTHONPATH=. python eval/epistemic_stress/run_epistemic_eval.py \\
  --out eval/epistemic_stress/results/latest.json \\
  --auto-label-with-openai \\
  --claim-annotate-n 20
```

`eval/epistemic_stress/results/` is gitignored.

## Output JSON

Top-level keys: `dataset_summary`, `answer_level_results`, `claim_level_results`,
`category_breakdown`, `score_separation`, `sample_cases`, `raw_jsonl`.

Do **not** commit API keys or raw logs if they contain sensitive content.
