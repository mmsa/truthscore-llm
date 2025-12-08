# Manual Experiment Guide

You don't need API calls! You can provide your own questions and answers.

## Option 1: Use Template File

### Step 1: Create Template

```bash
python3 -c "from experiments.run_manual_experiment import create_template_file; create_template_file()"
```

This creates `experiments/manual_answers_template.json` with all 50 prompts.

### Step 2: Fill in Your Answers

Edit the template file and add your answers for each method:

```json
{
  "prompt": "Does vitamin C prevent the common cold?",
  "vanilla": "No, vitamin C does not prevent the common cold...",
  "rag": "Research shows that vitamin C supplementation...",
  "self_consistency": "Vitamin C may help reduce duration...",
  "truthscore": "No, vitamin C does not prevent the common cold...",
  "ground_truth": {
    "answer": "No, vitamin C does not prevent colds",
    "is_correct": true
  }
}
```

### Step 3: Run Experiment

```bash
python3 -m experiments.run_manual_experiment experiments/manual_answers_template.json
```

## Option 2: Use Python Directly

```python
from experiments.run_manual_experiment import ManualExperimentRunner
from experiments.prompts import ALL_PROMPTS

# Your answers
answers = {
    "Does vitamin C prevent the common cold?": {
        "vanilla": "No, vitamin C does not prevent the common cold.",
        "rag": "Research indicates vitamin C does not prevent colds.",
        "self_consistency": "Vitamin C may help but doesn't prevent.",
        "truthscore": "No, vitamin C does not prevent the common cold."
    },
    # ... add more
}

# Run experiment
runner = ManualExperimentRunner()
results = runner.run_with_answers(ALL_PROMPTS[:10], answers)  # Use subset
annotated = runner.annotate_results(results)
summary = runner.summarize_results(annotated)
runner.print_summary_table(summary)
runner.save_results(annotated)
```

## Option 3: Use Existing Results

If you already have LLM outputs (from API calls, manual collection, etc.), just format them and use Option 1 or 2.

## What You Get

- TruthScore evaluation for all answers
- Outcome category annotations
- Summary table (Table 2) for your paper
- Full results JSON file

## No API Required!

This approach is perfect for:
- Using pre-generated LLM outputs
- Manual answer collection
- Testing with your own data
- Paper experiments where you already have answers

Just provide questions and answers - TruthScore will evaluate them automatically!

