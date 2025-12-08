# Quick Start: Running Real Experiments

## Step 1: Install OpenAI Package

```bash
pip install openai
```

Or install with the experiments extra:
```bash
pip install truthscore-llm[experiments]
```

## Step 2: Set Your API Key

```bash
export OPENAI_API_KEY='sk-your-key-here'
```

**Get your API key from:** https://platform.openai.com/api-keys

## Step 3: Verify Setup

```bash
python -m experiments.setup_api
```

You should see:
```
✓ OpenAI package is installed
✓ OPENAI_API_KEY is set
✓ API connection successful
```

## Step 4: Run the Experiment

### Full Experiment (50 prompts)

```bash
python -m experiments.run_experiment
```

**Note:** This makes ~450 API calls:
- 50 prompts × Vanilla LLM (1 call each) = 50 calls
- 50 prompts × RAG (1 call each) = 50 calls  
- 50 prompts × Self-Consistency (5 calls each) = 250 calls
- 50 prompts × TruthScore (1 call + scoring) = 50 calls

**Estimated cost:** $0.50 - $2.00 with GPT-4o-mini

### Test Run (5 prompts)

To test first, modify `run_experiment.py` or run:

```python
from experiments.run_experiment import ExperimentRunner
from experiments.prompts import ALL_PROMPTS

runner = ExperimentRunner()
results = runner.run_all_prompts(ALL_PROMPTS[:5])  # Test on 5 prompts
annotated = runner.annotate_results(results)
summary = runner.summarize_results(annotated)
runner.print_summary_table(summary)
runner.save_results(annotated)
```

## Step 5: View Results

Results are saved to:
- `experiments/results/experiment_results.json` - Full detailed results
- `experiments/results/experiment_summary.json` - Summary statistics

## Troubleshooting

### "OPENAI_API_KEY not set"
- Make sure you exported the environment variable
- Or set it in your shell: `export OPENAI_API_KEY='your-key'`

### "OpenAI package not installed"
- Run: `pip install openai`

### Rate Limits
- The code includes small delays, but if you hit limits, add more delay
- Or run in smaller batches

### Cost Concerns
- Start with 5-10 prompts to test
- Monitor usage at https://platform.openai.com/usage
- GPT-4o-mini is very affordable (~$0.15/$0.60 per 1M tokens)

## Next Steps

1. **Review Results**: Check `experiment_results.json` for actual answers
2. **Manual Annotation**: Review and annotate answers according to ground truth
3. **Generate Table**: Use summary statistics for your paper's Table 2
4. **Analysis**: Compare methods on different prompt categories

## For Your Paper

The experiment generates:
- **Table 2**: Outcome categories by inference method
- **Results**: Detailed responses for all 50 prompts
- **Statistics**: Counts and proportions for each category

You can use these directly in your paper!

