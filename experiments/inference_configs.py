"""
Inference configuration implementations for TruthScore experiments.

Implements four inference configurations:
1. Vanilla LLM decoding
2. Retrieval-Augmented Generation (RAG)
3. Self-consistency sampling (5 samples)
4. Truth Score inference
"""

import os
import time
from typing import List, Dict, Optional
from truthscore import TruthScorer

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Install with: pip install openai")


class InferenceConfig:
    """Base class for inference configurations."""
    
    def generate(self, prompt: str) -> Dict[str, any]:
        """
        Generate response for a given prompt.
        
        Returns:
            Dictionary with 'answer' and optional metadata
        """
        raise NotImplementedError


class VanillaLLM(InferenceConfig):
    """
    Vanilla LLM decoding - direct generation without augmentation.
    
    Uses OpenAI API with GPT-4o-mini for real inference.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize Vanilla LLM.
        
        Args:
            model_name: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            if not OPENAI_AVAILABLE:
                print("Warning: OpenAI package not installed. Using placeholder responses.")
            elif not self.api_key:
                print("Warning: OPENAI_API_KEY not set. Using placeholder responses.")
    
    def generate(self, prompt: str) -> Dict[str, any]:
        """
        Generate answer using vanilla LLM decoding.
        
        Returns real API response if available, otherwise placeholder.
        """
        if self.client is None:
            # Fallback to placeholder
            return {
                "answer": f"[PLACEHOLDER] This is a simulated vanilla LLM response to: {prompt}",
                "method": "vanilla",
                "model": self.model_name,
                "placeholder": True
            }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate, factual answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "method": "vanilla",
                "model": self.model_name,
                "placeholder": False,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "answer": f"[ERROR] Failed to generate response: {str(e)}",
                "method": "vanilla",
                "model": self.model_name,
                "error": str(e),
                "placeholder": True
            }


class RAG(InferenceConfig):
    """
    Retrieval-Augmented Generation.
    
    Retrieves relevant documents and conditions LLM generation on them.
    Note: This is a simplified RAG implementation. For production, integrate
    with a vector database (Pinecone, Weaviate, etc.).
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize RAG.
        
        Args:
            model_name: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def _retrieve_documents(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Placeholder implementation - replace with actual vector search.
        """
        # TODO: Integrate with vector database
        # For now, return placeholder documents
        return [
            f"Document 1 related to: {query}",
            f"Document 2 related to: {query}",
            f"Document 3 related to: {query}",
        ]
    
    def generate(self, prompt: str) -> Dict[str, any]:
        """
        Generate answer using RAG.
        
        Retrieves documents and conditions LLM generation on them.
        """
        # Retrieve relevant documents
        retrieved_docs = self._retrieve_documents(prompt, top_k=3)
        context = "\n\n".join([f"- {doc}" for doc in retrieved_docs])
        
        if self.client is None:
            return {
                "answer": f"[PLACEHOLDER] RAG response to: {prompt} (with {len(retrieved_docs)} retrieved docs)",
                "method": "rag",
                "model": self.model_name,
                "retrieved_docs": len(retrieved_docs),
                "placeholder": True
            }
        
        try:
            # Format prompt with context
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain enough information to answer the question, say so."""
            
            user_prompt = f"""Context:
{context}

Question: {prompt}

Answer based on the context above:"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "method": "rag",
                "model": self.model_name,
                "retrieved_docs": len(retrieved_docs),
                "placeholder": False,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"Error calling OpenAI API for RAG: {e}")
            return {
                "answer": f"[ERROR] Failed to generate RAG response: {str(e)}",
                "method": "rag",
                "model": self.model_name,
                "error": str(e),
                "placeholder": True
            }


class SelfConsistency(InferenceConfig):
    """
    Self-consistency sampling with 5 samples.
    
    Generates multiple answers and selects the most consistent one.
    """
    
    def __init__(self, num_samples: int = 5, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize Self-Consistency.
        
        Args:
            num_samples: Number of samples to generate (default: 5)
            model_name: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        """
        self.num_samples = num_samples
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def _compute_agreement(self, samples: List[str]) -> Dict[str, int]:
        """
        Compute agreement between samples.
        
        Simple implementation: counts exact matches.
        In production, could use semantic similarity.
        """
        # Count exact matches
        from collections import Counter
        return dict(Counter(samples))
    
    def generate(self, prompt: str) -> Dict[str, any]:
        """
        Generate answer using self-consistency sampling.
        
        Generates multiple samples and selects the most consistent one.
        """
        if self.client is None:
            samples = [
                f"[PLACEHOLDER] Sample {i} answer to: {prompt}"
                for i in range(self.num_samples)
            ]
            return {
                "answer": samples[0],
                "method": "self_consistency",
                "model": self.model_name,
                "samples": samples,
                "num_samples": self.num_samples,
                "placeholder": True
            }
        
        try:
            samples = []
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            # Generate multiple samples with higher temperature
            for i in range(self.num_samples):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,  # Higher temperature for diversity
                    max_tokens=200
                )
                
                answer = response.choices[0].message.content
                samples.append(answer)
                
                # Accumulate usage
                total_usage["prompt_tokens"] += response.usage.prompt_tokens
                total_usage["completion_tokens"] += response.usage.completion_tokens
                total_usage["total_tokens"] += response.usage.total_tokens
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
            
            # Compute agreement and select most common answer
            agreement = self._compute_agreement(samples)
            selected_answer = max(agreement.items(), key=lambda x: x[1])[0]
            
            return {
                "answer": selected_answer,
                "method": "self_consistency",
                "model": self.model_name,
                "samples": samples,
                "num_samples": self.num_samples,
                "agreement": agreement,
                "placeholder": False,
                "usage": total_usage
            }
        except Exception as e:
            print(f"Error calling OpenAI API for Self-Consistency: {e}")
            return {
                "answer": f"[ERROR] Failed to generate self-consistency response: {str(e)}",
                "method": "self_consistency",
                "model": self.model_name,
                "error": str(e),
                "placeholder": True
            }


class TruthScoreInference(InferenceConfig):
    """
    Truth Score inference.
    
    Uses TruthScore to evaluate and filter/refuse answers based on evidence.
    """
    
    def __init__(self, base_config: InferenceConfig, scorer: Optional[TruthScorer] = None):
        """
        Initialize Truth Score inference wrapper.
        
        Args:
            base_config: Base inference configuration (VanillaLLM, RAG, etc.)
            scorer: TruthScorer instance (creates default if None)
        """
        self.base_config = base_config
        self.scorer = scorer if scorer is not None else TruthScorer()
    
    def generate(self, prompt: str) -> Dict[str, any]:
        """
        Generate answer using base config, then evaluate with TruthScore.
        
        Returns answer only if TruthScore decision is ACCEPT or QUALIFIED.
        Otherwise returns a refusal.
        """
        # Generate answer using base configuration
        base_result = self.base_config.generate(prompt)
        answer = base_result["answer"]
        
        # Evaluate with TruthScore
        score_result = self.scorer.score(question=prompt, answer=answer)
        
        # Decision logic: refuse if REFUSE, otherwise return answer
        if score_result["decision"] == "REFUSE":
            return {
                "answer": f"I cannot provide a confident answer to this question based on available evidence.",
                "method": "truthscore",
                "base_method": base_result.get("method", "unknown"),
                "truth_score": score_result["truth_score"],
                "decision": "REFUSE",
                "refused": True,
                "score_details": score_result
            }
        else:
            return {
                "answer": answer,
                "method": "truthscore",
                "base_method": base_result.get("method", "unknown"),
                "truth_score": score_result["truth_score"],
                "decision": score_result["decision"],
                "refused": False,
                "score_details": score_result
            }

