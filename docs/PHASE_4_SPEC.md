# Phase 4 Spec — Production LLM Gateway Core

## Overview
Phase 4 wraps the core RAG services into an enterprise-grade async FastAPI Gateway with token authentication, token-bucket rate limiting, SSE streaming support, request tracing, and resilient error recovery.

## Architecture & Middleware Layers

```text
Client Request
    │
    ▼
[Request ID & Correlation Middleware]
    │
    ▼
[Bearer Token Authentication Middleware]
    │
    ▼
[Slowapi Rate Limiting Middleware]
    │
    ▼
[Gateway API Routes] ──> [Async HTTP Client] ──> [LLM / Provider Engine]
```

## Key Components

1. **Async Streaming (`StreamingResponse`):**
   - Implements Server-Sent Events (SSE) for token-by-token real-time streaming output to web and voice clients.

2. **Resilience & Retry Policies:**
   - Exponential backoff with jitter on HTTP 429/503 provider errors using `tenacity` library.
   - Configurable timeout limits per downstream model provider (e.g., 5s connect, 30s read).

3. **Endpoints:**
   - `POST /api/v1/gateway/chat/completions` (OpenAI-compatible request format).
