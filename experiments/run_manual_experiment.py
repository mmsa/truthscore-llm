"""
Manual experiment runner - use your own questions and answers.

This version allows you to provide answers manually or from your own sources,
without requiring API calls. Perfect for paper experiments where you already
have LLM outputs.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path

from experiments.prompts import ALL_PROMPTS, PROMPT_CATEGORIES
from experiments.annotation import OutcomeCategory, Annotator
from truthscore import TruthScorer


class ManualExperimentRunner:
    """Experiment runner for manual Q&A pairs."""
    
    def __init__(self, output_dir: str = "experiments/results"):
        """
        Initialize manual experiment runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scorer = TruthScorer()
        self.annotator = Annotator()
    
    def run_with_answers(
        self,
        prompts: List[str],
        answers: Dict[str, Dict[str, str]],
        ground_truth: Optional[Dict[str, any]] = None
    ) -> List[Dict]:
        """
        Run experiment with provided answers.
        
        Args:
            prompts: List of prompts/questions
            answers: Dictionary mapping prompt -> method -> answer
                Example: {
                    "What is the capital of France?": {
                        "vanilla": "Paris is the capital of France.",
                        "rag": "The capital of France is Paris.",
                        "self_consistency": "Paris.",
                        "truthscore": "Paris is the capital of France."
                    }
                }
            ground_truth: Optional ground truth information
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for prompt in prompts:
            result = {
                "prompt": prompt,
                "vanilla": {"answer": answers.get(prompt, {}).get("vanilla", "")},
                "rag": {"answer": answers.get(prompt, {}).get("rag", "")},
                "self_consistency": {"answer": answers.get(prompt, {}).get("self_consistency", "")},
                "truthscore": {"answer": answers.get(prompt, {}).get("truthscore", "")},
            }
            
            # Evaluate TruthScore for truthscore method
            truthscore_answer = result["truthscore"]["answer"]
            if truthscore_answer:
                score_result = self.scorer.score(question=prompt, answer=truthscore_answer)
                result["truthscore"]["truth_score"] = score_result["truth_score"]
                result["truthscore"]["decision"] = score_result["decision"]
                result["truthscore"]["score_details"] = score_result
            
            results.append(result)
        
        return results
    
    def run_from_file(self, input_file: str) -> List[Dict]:
        """
        Load prompts and answers from a JSON file.
        
        Expected format:
        [
            {
                "prompt": "Question here",
                "vanilla": "Answer here",
                "rag": "Answer here",
                "self_consistency": "Answer here",
                "truthscore": "Answer here"
            },
            ...
        ]
        """
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        prompts = [item["prompt"] for item in data]
        answers = {
            item["prompt"]: {
                "vanilla": item.get("vanilla", ""),
                "rag": item.get("rag", ""),
                "self_consistency": item.get("self_consistency", ""),
                "truthscore": item.get("truthscore", ""),
            }
            for item in data
        }
        
        return self.run_with_answers(prompts, answers)
    
    def annotate_results(self, results: List[Dict], ground_truth: Dict[str, any] = None) -> List[Dict]:
        """
        Annotate results with outcome categories.
        
        Args:
            results: List of result dictionaries
            ground_truth: Dictionary mapping prompts to ground truth info
        
        Returns:
            Annotated results
        """
        if ground_truth is None:
            ground_truth = {}
        
        annotated = []
        for result in results:
            prompt = result["prompt"]
            gt_info = ground_truth.get(prompt, {})
            
            annotated_result = result.copy()
            annotated_result["annotations"] = {}
            
            # Annotate each method
            for method in ["vanilla", "rag", "self_consistency", "truthscore"]:
                answer = result[method]["answer"]
                if not answer:
                    continue
                    
                is_refusal = self.annotator.detect_refusal(answer)
                is_hedged = self.annotator.detect_hedging(answer)
                
                category = self.annotator.annotate(
                    prompt=prompt,
                    answer=answer,
                    ground_truth=gt_info.get("answer"),
                    is_correct=gt_info.get("is_correct"),
                    is_refusal=is_refusal,
                    is_hedged=is_hedged
                )
                
                annotated_result["annotations"][method] = {
                    "category": category.value,
                    "is_refusal": is_refusal,
                    "is_hedged": is_hedged,
                }
            
            annotated.append(annotated_result)
        
        return annotated
    
    def summarize_results(self, annotated_results: List[Dict]) -> Dict:
        """
        Summarize experiment results.
        
        Args:
            annotated_results: List of annotated result dictionaries
        
        Returns:
            Summary statistics dictionary
        """
        methods = ["vanilla", "rag", "self_consistency", "truthscore"]
        categories = [cat.value for cat in OutcomeCategory]
        
        summary = {
            "total_prompts": len(annotated_results),
            "by_method": {},
            "by_category": {cat: {} for cat in categories}
        }
        
        # Count by method and category
        for method in methods:
            summary["by_method"][method] = {cat: 0 for cat in categories}
            
            for result in annotated_results:
                if method in result["annotations"]:
                    category = result["annotations"][method]["category"]
                    summary["by_method"][method][category] += 1
        
        # Count by category across all methods
        for category in categories:
            for method in methods:
                summary["by_category"][category][method] = summary["by_method"][method][category]
        
        return summary
    
    def save_results(self, results: List[Dict], filename: str = "manual_experiment_results.json"):
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def save_summary(self, summary: Dict, filename: str = "manual_experiment_summary.json"):
        """Save summary to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {filepath}")
    
    def print_summary_table(self, summary: Dict):
        """Print summary as a formatted table."""
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        methods = ["vanilla", "rag", "self_consistency", "truthscore"]
        categories = [cat.value for cat in OutcomeCategory]
        
        # Print header
        print(f"\n{'Method':<20} " + " ".join(f"{cat[:15]:<15}" for cat in categories))
        print("-" * 80)
        
        # Print rows
        for method in methods:
            row = f"{method:<20} "
            for category in categories:
                count = summary["by_method"][method][category]
                row += f"{count:<15} "
            print(row)
        
        print("\n" + "="*80)


def create_template_file(filename: str = "experiments/manual_answers_template.json"):
    """Create a template file for manual answers."""
    template = [
        {
            "prompt": "Does vitamin C prevent the common cold?",
            "vanilla": "Your vanilla LLM answer here",
            "rag": "Your RAG answer here",
            "self_consistency": "Your self-consistency answer here",
            "truthscore": "Your truthscore answer here",
            "ground_truth": {
                "answer": "Correct answer if known",
                "is_correct": True  # or False
            }
        }
    ]
    
    # Add all prompts
    for prompt in ALL_PROMPTS:
        template.append({
            "prompt": prompt,
            "vanilla": "",
            "rag": "",
            "self_consistency": "",
            "truthscore": "",
            "ground_truth": {
                "answer": "",
                "is_correct": None
            }
        })
    
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Template created at {filepath}")
    print(f"Fill in the answers and run: python -m experiments.run_manual_experiment")


def main():
    """Run manual experiment from file."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.run_manual_experiment <input_file.json>")
        print("\nTo create a template file:")
        print("  python -c 'from experiments.run_manual_experiment import create_template_file; create_template_file()'")
        return
    
    input_file = sys.argv[1]
    
    print(f"Loading answers from {input_file}...")
    runner = ManualExperimentRunner()
    
    results = runner.run_from_file(input_file)
    
    print(f"Annotating {len(results)} results...")
    annotated_results = runner.annotate_results(results)
    
    print("Summarizing results...")
    summary = runner.summarize_results(annotated_results)
    
    runner.print_summary_table(summary)
    
    runner.save_results(annotated_results)
    runner.save_summary(summary)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()

