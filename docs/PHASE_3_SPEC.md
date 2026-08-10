# Phase 3 Spec — Empirical RAG & LLM Evaluation

## Overview
Phase 3 implements an automated evaluation harness to quantitatively measure retrieval accuracy and generation quality across different retrieval strategies (Dense vs Hybrid vs Re-ranked).

## Benchmark Dataset Schema (`tests/eval_dataset.json`)

```json
[
  {
    "id": "eval_001",
    "question": "What is the continuous batching mechanism in ApexGuard?",
    "expected_answer": "Continuous batching dynamically inserts new requests into running inference batches at the iteration level.",
    "ground_truth_doc": "architecture_v1.pdf",
    "ground_truth_chunks": ["chunk_12", "chunk_13"]
  }
]
```

## Evaluated Metrics & Formulas

1. **Retrieval Metrics:**
   - **Precision@K:** $\frac{|\text{Retrieved Chunks} \cap \text{Ground Truth Chunks}|}{K}$
   - **Recall@K:** $\frac{|\text{Retrieved Chunks} \cap \text{Ground Truth Chunks}|}{|\text{Ground Truth Chunks}|}$
   - **Mean Reciprocal Rank (MRR):** $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$

2. **Generation Quality Metrics:**
   - **Context Relevance:** Ratio of retrieved context sentences directly relevant to the user question.
   - **Faithfulness:** Proportion of claims in the generated answer directly supported by the context.
   - **Answer Correctness:** Semantic similarity between generated output and `expected_answer`.

## Execution Harness

- Script: `python -m tests.run_eval`
- Generates comparative Markdown & JSON evaluation reports saved in `docs/eval_reports/`.
