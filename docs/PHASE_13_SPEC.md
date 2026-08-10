# Phase 13 Spec — High-Throughput Model Serving (vLLM)

## Overview
Phase 13 replaces standard local Ollama inference with **vLLM**, leveraging PagedAttention, continuous batching, and KV cache optimization for enterprise-scale GPU serving.

## Performance Profiling Metrics

1. **Continuous Batching:** Dynamically schedules arriving requests at iteration granularity instead of sequence granularity.
2. **PagedAttention KV Cache:** Manages Key-Value memory pages efficiently to prevent GPU memory fragmentation.
3. **Quantization:** Benchmark AWQ (Activation-aware Weight Quantization) / FP8 vs FP16 precision throughput (tokens/sec) vs memory footprint.

## Integration Interface

- Exposes vLLM's OpenAI-compatible HTTP server (`vllm.entrypoints.openai.api_server`) hooked directly into the Phase 4 FastAPI Gateway.
