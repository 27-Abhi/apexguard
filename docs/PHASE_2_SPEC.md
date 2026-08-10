# Phase 2 Spec — Advanced Hybrid Retrieval & Re-ranking

## Overview
Phase 2 upgrades the basic Phase 1 dense retrieval pipeline into a production-grade multi-stage hybrid search system combining dense vector search with sparse lexical search (BM25) and local cross-encoder re-ranking.

## Architecture

```text
User Query
    │
    ├───> Dense Vector Search (Qdrant Cosine Distance) ──────┐
    │                                                        ├──> Reciprocal Rank Fusion (RRF) ──> Top 10 ──> FlashRank Re-ranker ──> Top 3 ──> LLM
    └───> Sparse Lexical Search (FastEmbed Sparse / BM25) ───┘
```

## Core Modules & Implementation Details

1. **Sparse Embedding & Lexical Search:**
   - **Engine:** `FastEmbed` Sparse text embeddings (`Qdrant/bm25`).
   - **Purpose:** Capture exact keyword matches, technical symbols, acronyms, and product codes that dense semantic vectors often miss.

2. **Reciprocal Rank Fusion (RRF):**
   - **Formula:** $RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$ where $k=60$.
   - Combines rank lists from dense and sparse vector queries into a single balanced relevance ranking.

3. **Cross-Encoder Re-ranking:**
   - **Library:** `FlashRank` (Ultra-lite ONNX-based local re-ranking).
   - Evaluates joint query-chunk attention pairs to re-score the top 10 RRF candidates and trim to the final Top 3 high-confidence context chunks.

## API Endpoint Expansion

- `POST /api/v1/query/hybrid`
  - Accepts `query`, `top_k` (default 3), `fusion_weight` (dense vs sparse weight), and `enable_rerank` (boolean flag).
