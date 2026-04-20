"""
Main experiment runner for TruthScore evaluation.

Runs experiments comparing four inference configurations:
1. Vanilla LLM decoding
2. Retrieval-Augmented Generation (RAG)
3. Self-consistency sampling (5 samples)
4. Truth Score inference
"""

import json
from typing import List, Dict
from pathlib import Path

from experiments.prompts import ALL_PROMPTS, PROMPT_CATEGORIES
from experiments.inference_configs import (
    VanillaLLM, RAG, SelfConsistency, TruthScoreInference
)
from experiments.annotation import OutcomeCategory, Annotator


class ExperimentRunner:
    """Main experiment runner."""
    
    def __init__(self, output_dir: str = "experiments/results"):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize inference configurations
        self.vanilla = VanillaLLM()
        self.rag = RAG()
        self.self_consistency = SelfConsistency(num_samples=5)
        
        # Truth Score inference wraps vanilla LLM by default
        self.truthscore = TruthScoreInference(self.vanilla)
        
        self.annotator = Annotator()
    
    def run_single_prompt(self, prompt: str) -> Dict:
        """
        Run all inference configurations on a single prompt.
        
        Args:
            prompt: The question/prompt to evaluate
        
        Returns:
            Dictionary with results from all configurations
        """
        results = {
            "prompt": prompt,
            "vanilla": self.vanilla.generate(prompt),
            "rag": self.rag.generate(prompt),
            "self_consistency": self.self_consistency.generate(prompt),
            "truthscore": self.truthscore.generate(prompt),
        }
        
        return results
    
    def run_all_prompts(self, prompts: List[str] = None) -> List[Dict]:
        """
        Run experiment on all prompts.
        
        Args:
            prompts: List of prompts (uses ALL_PROMPTS if None)
        
        Returns:
            List of results dictionaries
        """
        if prompts is None:
            prompts = ALL_PROMPTS
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
            result = self.run_single_prompt(prompt)
            results.append(result)
        
        return results
    
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
                category = result["annotations"][method]["category"]
                summary["by_method"][method][category] += 1
        
        # Count by category across all methods
        for category in categories:
            for method in methods:
                summary["by_category"][category][method] = summary["by_method"][method][category]
        
        return summary
    
    def save_results(self, results: List[Dict], filename: str = "experiment_results.json"):
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def save_summary(self, summary: Dict, filename: str = "experiment_summary.json"):
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


def main():
    """Run the experiment."""
    print("Starting TruthScore Experiment")
    print("="*80)
    
    runner = ExperimentRunner()
    
    # Run experiment on all prompts
    print(f"\nRunning experiment on {len(ALL_PROMPTS)} prompts...")
    results = runner.run_all_prompts()
    
    # Annotate results (using placeholder ground truth)
    print("\nAnnotating results...")
    annotated_results = runner.annotate_results(results)
    
    # Summarize
    print("\nSummarizing results...")
    summary = runner.summarize_results(annotated_results)
    
    # Print summary table
    runner.print_summary_table(summary)
    
    # Save results
    runner.save_results(annotated_results)
    runner.save_summary(summary)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()

