# ApexGuard — Master Documentation & 16-Phase Implementation Roadmap

> **Platform:** ApexGuard — Real-Time LLM Gateway & AI Agent Platform  
> **Engineering Focus:** Production-Grade AI Systems, RAG Engineering, LLMOps, Model Serving, & Speech ML  
> **Target Alignment:** 2026 AI Engineering & AI Infrastructure Roles

---

## 1. Candidate Skill Matrix & Gap Analysis

Based on industry signals for AI Engineering in 2026, candidates must demonstrate empirical software craftsmanship, context engineering, evaluation benchmarks, and cloud operational capabilities.

### Profile Strengths (Baseline Foundation)
- **Languages & Frameworks:** Python, FastAPI, AsyncIO, WebSockets, REST APIs.
- **DevOps & Infrastructure:** Docker, Kubernetes, Linux, Production Debugging.
- **AI Domain:** Conversational AI, Voice Pipelines (STT → LLM → TTS), Enterprise AI integrations, Cost/Latency Optimization.

### Target Gap Closures in ApexGuard
- **RAG Architecture:** Hybrid search (Dense + Sparse), Re-ranking, Context optimization.
- **Vector Databases:** Qdrant collection schemas, payload indexing, Cosine distance matching.
- **Evaluation Frameworks:** Retrieval Precision, Recall, MRR, Context Relevance, Answer Faithfulness.
- **LLM Gateway Engineering:** Async rate-limiting, Intelligent routing (cost/latency trade-offs), Error handling, Retries.
- **State Management & Caching:** Redis session memory, conversation history, semantic response caching.
- **Deterministic Tool Calling:** Agent tool selection, argument validation, timeout handling.
- **LLMOps & Observability:** Langfuse tracing, OpenTelemetry metrics, TTFT (Time-To-First-Token) monitoring.
- **AI Security & Guardrails:** PII detection/redaction, max request size validation, prompt injection defenses.
- **Cloud Infrastructure & Deployment:** AWS EC2, S3, ECR, IAM roles, GitHub Actions CI/CD pipelines.
- **Inference Optimization:** vLLM, continuous batching, KV caching, quantization.
- **Speech ML Specialization:** Hugging Face ASR model evaluation (WER/CER), LoRA/PEFT fine-tuning.

---

## 2. Global System Architecture

```text
                    Client (Text / Web / Voice)
                                │
                                ▼
                       ┌─────────────────┐
                       │  FastAPI Gateway │
                       │   Async / Auth   │
                       └────────┬────────┘
                                │
                        ┌───────▼────────┐
                        │ Session/Router │
                        │     Redis      │
                        └───────┬────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
   Simple Query            RAG Pipeline          Tool Calling
  (Local / Small)      (Hybrid + Re-ranking)       (APIs / Calc)
         │                      │                      │
         │                   Qdrant                    │
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                ▼
                          Model Serving
                        (Ollama / vLLM)
                                │
                                ▼
                         Streaming / Voice
                        (STT -> RAG -> TTS)
```

**Cross-Cutting Concerns Across All Layers:**
- **Evaluation:** Automated quality metrics on retrieval and generation.
- **Observability:** Distributed tracing, latency breakdown, token counting.
- **Security:** Inbound/outbound guardrails & PII scrubbing.
- **Cost Tracking:** Per-model and per-request cost accounting.
- **CI/CD:** Automated testing, containerization, and AWS deployment.

---

## 3. Comprehensive Phase-by-Phase Roadmap

### PHASE 1 — Strong RAG Foundation (~8–12 hrs)
- **Status:** **Completed & Built**
- **Flow:** Documents → Ingestion → Chunking → FastEmbed → Qdrant Vector DB → Retriever → Ollama LLM → Answer + Sources.
- **Deliverable:** Working RAG API with FastAPI, Qdrant client, and local FastEmbed vectors.

### PHASE 2 — Advanced Hybrid Retrieval & Re-ranking (~8–12 hrs)
- **Flow:** Query → Dense Search (Qdrant) + Sparse Search (BM25) → Fusion (RRF) → Top 10 → Re-ranker (FlashRank) → Top 3 → LLM.
- **Deliverable:** Multi-stage retrieval engine eliminating single-vector search limitations.

### PHASE 3 — Empirical RAG & LLM Evaluation (~8 hrs)
- **Dataset:** Curated benchmark containing `(question, expected_answer, relevant_document, relevant_chunk)`.
- **Metrics:** Retrieval Precision, Recall, MRR, Context Relevance, Answer Correctness, Faithfulness, Total Latency.
- **Deliverable:** Automated benchmark runner script outputting quality telemetry reports.

### PHASE 4 — Production LLM Gateway Core (~8–10 hrs)
- **Features:** Async request processing, SSE streaming, rate limiting, request ID tracing, configurable timeouts, retries, and health checks.
- **Deliverable:** Resilient FastAPI gateway sitting in front of model providers.

### PHASE 5 — Intelligent Model Routing (~5–8 hrs)
- **Flow:** Query Classifier → Simple Queries (Local/Small Model) vs Complex/Knowledge Queries (RAG / Larger Model).
- **Deliverable:** Dynamic cost-aware router with empirical benchmarking on cost vs latency vs accuracy.

### PHASE 6 — Redis State & Response Caching (~5–8 hrs)
- **Features:** User session memory, conversation history windowing, distributed rate limiting, and exact/semantic response caching.
- **Deliverable:** Sub-10ms response caching for frequent queries and persistent multi-turn chat sessions.

### PHASE 7 — Agentic Tool Calling (~6–8 hrs)
- **Tools:** `search_knowledge()`, `get_document()`, `calculator()`, `external_api()`.
- **Features:** Strict Pydantic input schemas, execution error handling, retry policies, and final synthesis.
- **Deliverable:** Autonomous agent execution engine with deterministic tools.

### PHASE 8 — Observability & LLMOps (~5–7 hrs)
- **Metrics:** Time-to-First-Token (TTFT), retrieval latency, tool execution time, total end-to-end latency, prompt/completion token usage.
- **Integration:** Langfuse tracing & OpenTelemetry span context propagation.

### PHASE 9 — AI Guardrails & Security (~5–7 hrs)
- **Protections:** PII detector/redactor (regex + pattern scrubbing), request payload cap, rate limiter, approved tool whitelist, prompt injection filters.
- **Deliverable:** Inbound and outbound gateway sanitization middleware.

### PHASE 10 — Docker Compose & Automated CI/CD (~5–7 hrs)
- **Containers:** Gateway, Redis, Qdrant, Ollama.
- **Pipeline:** GitHub Actions running Pytest, lint checks, and container image builds.

### PHASE 11 — AWS Cloud Infrastructure (~8–12 hrs)
- **Services:** IAM roles, EC2 compute, ECR image repository, S3 document storage, CloudWatch logs, VPC security groups.
- **Strategy:** Free-tier local model inference combined with AWS host deployment.

### PHASE 12 — Automated AWS CI/CD Pipeline (~4–6 hrs)
- **Workflow:** Push to `main` → GitHub Actions → Build Docker Image → Push to ECR → Deploy to EC2 instance over SSH.

### PHASE 13 — High-Throughput Model Serving (vLLM) (~8–12 hrs)
- **Learnings:** Continuous batching, PagedAttention KV cache, GPU memory management, tensor parallelism, quantization profiling.

### PHASE 14 — Speech ML & ASR Specialization (~10–15 hrs)
- **Focus:** Hugging Face Transformers & Datasets for ASR model evaluation (Word Error Rate - WER, Character Error Rate - CER), plus LoRA/PEFT fine-tuning on speech domain samples.

### PHASE 15 — Voice Pipeline Integration (~6–8 hrs)
- **Flow:** Audio Input → ASR (Whisper) → ApexGuard Gateway → RAG/Tools → TTS → Audio Output.
- **Deliverable:** Full real-time streaming voice agent over WebSockets.

### PHASE 16 — GCP Infrastructure Mapping (~5–8 hrs)
- **Mapping:** AWS EC2 → Compute Engine, S3 → Cloud Storage, EKS → GKE, CloudWatch → Cloud Logging, ECR → Artifact Registry.

---

## 4. Milestone Timeline & Production Portfolio Strategy

| Milestone | Included Phases | Hours | Portfolio Capability Claim |
| :--- | :--- | :---: | :--- |
| **Milestone 1** | Phases 1–3 | **30h** | *"Built and evaluated a high-precision hybrid RAG engine with empirical metrics."* |
| **Milestone 2** | Phases 4–7 | **58h** | *"Engineered a real-time LLM Gateway with dynamic routing, Redis caching, and agent tool execution."* |
| **Milestone 3** | Phases 8–10 | **76h** | *"Implemented production LLMOps observability, guardrails, and containerized CI/CD pipelines."* |
| **Milestone 4** | Phases 11–16 | **121h** | *"Deployed production AI infrastructure on AWS, benchmarked vLLM serving, and conducted Speech ML evaluation."* |

---

## 5. Free-First Open Source Tech Stack

| Layer | Component | Choice |
| :--- | :--- | :--- |
| **API Gateway** | Web Framework | FastAPI + Uvicorn |
| **Vector DB** | Vector Storage | Qdrant (Local / Container) |
| **Embedding Engine** | Embeddings | FastEmbed / BAAI bge-small-en-v1.5 |
| **Re-ranker** | Cross-Encoder | FlashRank |
| **LLM Inference** | Model Host | Ollama / vLLM |
| **State & Cache** | In-Memory Data Store | Redis |
| **Observability** | LLM Tracing | Langfuse |
| **DevOps & Cloud** | CI & Cloud | Docker, GitHub Actions, AWS Free Tier (EC2/S3) |
| **Speech ML** | Audio Models | Hugging Face Transformers, Whisper |
