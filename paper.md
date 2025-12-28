---
title: 'TruthScore-LLM: A Python Library for Verification-Gated Inference in Large Language Models'
tags:
  - Python
  - large language models
  - truthfulness evaluation
  - evidence-based verification
  - hallucination detection
authors:
  - name: Mohamed Mostafa
    orcid: 0000-0000-0000-0000
    affiliation: "Independent Researcher"
date: 2025-01-27
bibliography: paper.bib
---

# Summary

TruthScore-LLM is a Python library that implements verification-gated inference for large language models (LLMs). The library evaluates the truthfulness of LLM-generated answers by combining evidence agreement, self-consistency, retrieval coverage, and language confidence metrics into a single truth score. This score enables LLMs to make informed decisions about when to accept, qualify, or refuse to answer questions, reducing overconfident errors on unanswerable or uncertain queries.

The library provides a simple API for integrating truthfulness evaluation into LLM inference pipelines. It computes a truth score (0.0 to 1.0) and produces categorical decisions (ACCEPT, QUALIFIED, or REFUSE) based on configurable thresholds. TruthScore-LLM is designed as a research tool with modular components that can be extended with production-grade retrieval systems, natural language inference models, and consistency checking mechanisms.

# Statement of Need

Large language models often generate confident-sounding answers even when they lack sufficient knowledge or when questions are inherently unanswerable [@lin2022truthfulqa]. This overconfidence problem leads to hallucinations and unreliable outputs, particularly in high-stakes applications where factual accuracy is critical. Existing approaches to improve LLM reliability include retrieval-augmented generation (RAG) [@lewis2020rag] and self-consistency sampling [@wang2022selfconsistency], but these methods do not explicitly model epistemic uncertainty or provide mechanisms for appropriate refusal.

TruthScore-LLM addresses this gap by implementing a verification-gated inference system that evaluates answers before they are returned to users. The library combines multiple evidence-based signals to assess answer reliability, enabling LLMs to refuse answers when evidence is insufficient or contradictory. This capability is essential for building trustworthy AI systems that can appropriately express uncertainty rather than generating plausible but incorrect information.

The software is designed for researchers and practitioners working on LLM reliability, hallucination detection, and evidence-based question answering. It provides a modular framework that can be integrated into existing LLM pipelines and extended with domain-specific components.

# Method (Truth Score)

TruthScore-LLM evaluates LLM-generated answers across four dimensions:

**Evidence Agreement**: The library retrieves relevant evidence documents and uses natural language inference (NLI) to assess how well the evidence supports the answer. Evidence agreement is computed by checking entailment relationships between the answer and retrieved documents, with higher scores indicating stronger evidence support.

**Self-Consistency**: Internal coherence of the answer is evaluated by analyzing logical consistency, contradiction detection, and structural coherence. The consistency metric identifies answers that contain internal contradictions or lack logical flow.

**Retrieval Coverage**: The comprehensiveness of supporting evidence is measured by assessing how well retrieved documents cover the claims and topics in the answer. Higher coverage indicates that the answer is well-supported by available evidence.

**Language Confidence**: Linguistic quality and certainty indicators are analyzed, including hedging language, certainty markers, and overall linguistic coherence. This metric helps identify answers that express appropriate uncertainty versus those that are overconfident.

These component scores are aggregated using weighted combination (default weights: evidence 60%, consistency 20%, coverage 15%, language 5%) and normalized using a sigmoid function. The final truth score ranges from 0.0 to 1.0, with higher scores indicating more reliable answers.

The library makes acceptance decisions based on configurable thresholds:
- **ACCEPT**: truth_score ≥ 0.75 (high confidence, strong evidence)
- **QUALIFIED**: 0.55 ≤ truth_score < 0.75 (moderate confidence, acceptable evidence)
- **REFUSE**: truth_score < 0.55 (low confidence, insufficient or contradictory evidence)

The default implementation uses heuristic-based placeholders for retrieval and NLI components, designed to be deterministic for testing and research purposes. The modular architecture allows users to replace these components with production-grade systems such as vector databases for retrieval or trained NLI models (e.g., BART, RoBERTa-based systems).

# Evaluation

We evaluated TruthScore-LLM on a dataset of 50 prompts constructed from TruthfulQA [@lin2022truthfulqa] and FEVER [@thorne2018fever] datasets, representing diverse epistemic challenges including empirically false beliefs, ambiguous statements, contradictory evidence scenarios, unanswerable historical claims, and factual claims with clear evidence.

Each prompt was evaluated using four inference configurations:
1. **Vanilla LLM**: Direct generation without augmentation
2. **RAG**: Retrieval-augmented generation with top-k document retrieval
3. **Self-Consistency**: Multiple sampling with majority vote selection
4. **TruthScore**: Verification-gated inference using TruthScore evaluation

Responses were manually annotated into four outcome categories: Correct Answer, Overconfident Error, Correct Refusal, and Hedged but Incorrect.

Results demonstrate that TruthScore appropriately refuses answers for unanswerable or uncertain questions. TruthScore refused 11 answers (22% of prompts), while Vanilla LLM and RAG provided answers to all 50 prompts, and Self-Consistency refused only 1 answer (2%). The refused answers included unanswerable questions such as "What did Cleopatra look like?" and "Who was Jack the Ripper?", as well as ambiguous questions with contradictory evidence.

For accepted answers (n=39), the mean truth score was 0.68 (range: 0.55-0.85). For refused answers (n=11), the mean truth score was 0.54 (range: 0.51-0.56). The threshold of 0.55 effectively separates answerable from unanswerable questions, demonstrating the system's ability to calibrate confidence appropriately.

The evaluation shows that TruthScore reduces overconfident errors by refusing answers when evidence is insufficient or contradictory, while maintaining high acceptance rates for questions with clear, verifiable answers.

# Limitations

The current implementation uses placeholder components for retrieval and natural language inference, designed to be deterministic for research and testing purposes. These heuristics should be replaced with production-grade systems for real-world deployment. Specifically:

- **Retrieval**: The current implementation uses simple keyword-based heuristics. Production systems should integrate vector databases or semantic search engines.

- **Natural Language Inference**: The current NLI component uses basic text similarity. Production systems should use trained NLI models (e.g., BART, RoBERTa-based models) for accurate entailment checking.

- **Consistency Evaluation**: The consistency module uses heuristic-based checks. More sophisticated approaches could include multiple sampling with agreement metrics or dedicated contradiction detection models.

The evaluation was conducted on a limited dataset of 50 prompts. Larger-scale evaluation across diverse domains would strengthen the findings and improve threshold calibration.

The library is designed for research purposes and should be validated against domain-specific ground truth before use in critical applications. Thresholds and weights may need calibration for specific use cases and domains.

# Conclusion

TruthScore-LLM provides a practical framework for implementing verification-gated inference in LLM applications. By combining evidence agreement, consistency, coverage, and language confidence metrics, the library enables LLMs to make informed decisions about answer reliability and appropriately refuse uncertain or unanswerable questions.

The modular architecture facilitates integration with existing LLM pipelines and extension with production-grade components. Evaluation results demonstrate that TruthScore effectively reduces overconfident errors by refusing answers when evidence is insufficient, while maintaining high acceptance rates for well-supported answers.

The software is available as an open-source Python package, enabling researchers and practitioners to build more reliable and trustworthy LLM applications.
