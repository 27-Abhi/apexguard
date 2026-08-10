# Phase 8 Spec — Observability & LLMOps (Langfuse & OpenTelemetry)

## Overview
Phase 8 integrates end-to-end telemetry, span tracing, latency breakdown, and token cost profiling using **Langfuse** and **OpenTelemetry**.

## Tracing Hierarchy

```text
[Span: Gateway Request]
   ├── [Span: Authentication & Rate Limit]
   ├── [Span: Redis Cache Lookup]
   ├── [Span: Hybrid Retrieval]
   │      ├── [Child Span: Qdrant Dense Vector Search]
   │      ├── [Child Span: FastEmbed Sparse Search]
   │      └── [Child Span: FlashRank Re-ranking]
   ├── [Span: LLM Generation]
   │      ├── Time-to-First-Token (TTFT)
   │      └── Total Completion Tokens
   └── [Span: Total Response Assembly]
```

## Tracked Telemetry Metrics
- **TTFT (ms):** Latency from request start to first streamed chunk.
- **Tokens/Sec:** Generation throughput.
- **Cost ($):** Estimated dollar cost based on model prompt/completion token pricing.
