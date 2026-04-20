# TruthScore Experiment Framework

This directory contains the experimental framework for evaluating TruthScore against various inference configurations.

## Setup

### 1. Install Dependencies

```bash
# Install OpenAI package
pip install openai

# Or install with experiments extra
pip install truthscore-llm[experiments]
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Verify Setup

```bash
python -m experiments.setup_api
```

This will check:
- OpenAI package installation
- API key configuration
- API connectivity

## Experiment Design

We constructed a set of 50 prompts from TruthfulQA and FEVER representing:

- **Empirically false popular beliefs** (10 prompts)
- **Ambiguous factual statements** (10 prompts)
- **Contradictory evidence scenarios** (10 prompts)
- **Unanswerable historical claims** (10 prompts)
- **Factual claims with evidence** (10 prompts)

Each query is submitted to four inference configurations:

1. **Vanilla LLM decoding** - Direct generation using GPT-4o-mini
2. **Retrieval-Augmented Generation (RAG)** - Retrieves relevant documents and conditions generation
3. **Self-consistency sampling** - Generates 5 samples and selects most consistent
4. **Truth Score inference** - Uses TruthScore to evaluate and filter/refuse answers

## Outcome Categories

Responses are coded according to epistemic outcome categories:

- **Correct Answer** - Answer is correct and confident
- **Overconfident Error** - Answer is wrong but confident
- **Correct Refusal** - Answer appropriately refuses when uncertain
- **Hedged but Incorrect** - Answer is wrong but contains hedging language

## Usage

### Running the Experiment

```bash
python -m experiments.run_experiment
```

This will:
1. Run all 50 prompts through all 4 inference configurations
2. Make real API calls to GPT-4o-mini
3. Annotate results with outcome categories
4. Generate summary statistics
5. Save results to `experiments/results/`

**Note:** This will make ~200 API calls (50 prompts × 4 methods). With self-consistency, it's ~450 calls (50 × 1 + 50 × 1 + 50 × 5 + 50 × 1). Monitor your API usage!

### Output Files

- `experiment_results.json` - Full results for all prompts with API responses
- `experiment_summary.json` - Summary statistics by method and category

### Running on a Subset

To test on a smaller subset first:

```python
from experiments.run_experiment import ExperimentRunner
from experiments.prompts import ALL_PROMPTS

runner = ExperimentRunner()
results = runner.run_all_prompts(ALL_PROMPTS[:5])  # Test on 5 prompts
```

## Cost Estimation

Using GPT-4o-mini:
- Input: ~$0.15 per 1M tokens
- Output: ~$0.60 per 1M tokens

For 50 prompts × 4 methods:
- Estimated cost: $0.50 - $2.00 (depending on response lengths)

## Customization

### Using Different Models

```python
from experiments.inference_configs import VanillaLLM

# Use GPT-4 instead
vanilla = VanillaLLM(model_name="gpt-4")
```

### Custom API Key

```python
from experiments.inference_configs import VanillaLLM

vanilla = VanillaLLM(api_key="your-custom-key")
```

### Adjusting Self-Consistency Samples

```python
from experiments.inference_configs import SelfConsistency

# Use 10 samples instead of 5
sc = SelfConsistency(num_samples=10)
```

## Manual Annotation

For accurate results, manual annotation is recommended:

1. Review each generated answer in `experiment_results.json`
2. Determine ground truth (if known)
3. Classify according to outcome categories
4. Update ground truth dictionary
5. Re-run annotation

## Results Analysis

The summary table shows counts by method and category:

```
Method                Correct Answer   Overconfident Error  Correct Refusal  Hedged but Incorrect
vanilla               15               20                    5                10
rag                   18               18                    7                7
self_consistency      20               15                    8                7
truthscore            22               12                    12               4
```

This allows comparison of:
- **Correct Answer Rate** - Higher is better
- **Overconfident Error Rate** - Lower is better
- **Correct Refusal Rate** - Higher is better (for uncertain queries)
- **Hedged but Incorrect Rate** - Lower is better

## Paper Integration

Results from this experiment can be used to generate Table 2 in the paper, showing observed behavior across inference configurations.

## Troubleshooting

### API Rate Limits

If you hit rate limits, add delays:

```python
import time
time.sleep(1)  # Wait 1 second between calls
```

### API Errors

Check your API key and quota:
- Verify key is correct
- Check OpenAI dashboard for usage/quota
- Ensure you have credits

### Placeholder Responses

If you see `[PLACEHOLDER]` in responses:
- Check that `OPENAI_API_KEY` is set
- Verify OpenAI package is installed
- Run `python -m experiments.setup_api` to diagnose
