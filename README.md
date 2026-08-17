# 🛡️ ApexGuard — Knowledge Transfer (KT) & Architecture Manual

Welcome to **ApexGuard**! This document serves as the complete **Knowledge Transfer (KT) manual** and operational guide for engineers, developers, and newcomers joining the project.

---

## 📚 1. Knowledge Transfer (KT) Summary: What Has Been Done?

ApexGuard has evolved into a **Phase 3 Empirical RAG & LLM Evaluation Platform**. Below is a comprehensive breakdown grounded directly in the codebase implementation:


### 1. Multi-Format Ingestion & Chunking Strategies ([`ingestion_service.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/app/services/ingestion_service.py))

ApexGuard supports **4 distinct chunking strategies** configured via the `ChunkingStrategy` enum:

#### Strategies & Code Implementations:
1. **`recursive` (Recursive Character Chunking - Default):**
   * **Mechanism:** Recursively tries separators `["\n\n", "\n", ". ", " ", ""]` to keep related paragraphs and sentences intact without exceeding `chunk_size`.
   * **Best For:** General prose, articles, and documentation.
2. **`fixed` (Fixed-Size Window Chunking):**
   * **Mechanism:** Hard character index windowing (`start` to `start + chunk_size`) stepping by `(chunk_size - overlap)`.
   * **Best For:** Uniform tabular logs or fixed-length code blocks.
3. **`sentence` (Sentence-Boundary Chunking):**
   * **Mechanism:** Regex-splits text into full sentences (`(?<=[.!?])\s+`) and accumulates them up to `chunk_size`.
   * **Best For:** Legal or medical texts where sentence context must never be sliced mid-phrase.
4. **`semantic` (Cosine Similarity-Based Semantic Chunking):**
   * **Mechanism:** Computes cosine similarity between embeddings of adjacent sentences (`np.dot(v1, v2)/(||v1||*||v2||)`). Splits when similarity falls below `similarity_threshold=0.70` or `chunk_size` is exceeded.
   * **Best For:** Multi-topic documents with soft topic transitions.

#### 💡 Concrete Chunking Example:
* **Raw Input Text (Length = 120 chars):**
  > `"ApexGuard protects LLM gateways. It provides hybrid vector search. Vector search uses Qdrant. Re-ranking uses FlashRank."`
* **Execution with `chunk_size=70`, `overlap=15`:**
  * **`fixed` Result:**
    * Chunk 1: `"ApexGuard protects LLM gateways. It provides hybrid vector search."` (Length 67)
    * Chunk 2: `"provides hybrid vector search. Vector search uses Qdrant. Re-ranking"` (Includes 15 char overlap)
  * **`recursive` Result:**
    * Chunk 1: `"ApexGuard protects LLM gateways. It provides hybrid vector search."` (Splits on `. ` boundary)
    * Chunk 2: `"Vector search uses Qdrant. Re-ranking uses FlashRank."` (Preserves sentence structure)

---

### 2. Retrieval Mechanisms: Dense Search vs. Sparse Search ([`rag_service.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/app/services/rag_service.py) & [`vector_service.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/app/services/vector_service.py))

ApexGuard implements a **Hybrid Retrieval Engine** combining Dense and Sparse vector search.

| Feature | Dense Vector Search | Sparse Vector Search (BM25) |
| :--- | :--- | :--- |
| **Model** | `FastEmbed` (`BAAI/bge-small-en-v1.5`) | `FastEmbed` (`Qdrant/bm25`) |
| **Representation** | 384-dimensional dense floating-point array | High-dimensional sparse term-frequency indices |
| **Matching Logic** | Semantic Similarity (Cosine Distance) | Exact Keyword / Token Match (Term Frequency) |
| **Strengths** | Understands synonyms, paraphrasing, and intent (e.g., *"error handling"* matches *"exception catching"*) | Finds exact terms, acronyms, product IDs (e.g., *"CVE-2024-1234"* or *"API_V1_STR"*) |
| **Weaknesses** | Struggles with rare out-of-vocabulary keywords or precise codes | Fails when query uses synonyms without exact term matches |

#### 🔀 Reciprocal Rank Fusion (RRF):
Dense and Sparse candidate lists are merged using RRF scoring:
$$\text{RRF Score}(d) = \frac{w_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{60 + \text{rank}_{\text{sparse}}(d)}$$
Where $w_{\text{dense}} = \text{fusion\_weight}$ and $w_{\text{sparse}} = 1.0 - \text{fusion\_weight}$.

---

### 3. Re-Ranking Engine: FlashRank & Alternatives ([`rag_service.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/app/services/rag_service.py#L81-L107))

Vector retrieval (Cosine/BM25) evaluates candidate chunks **independently**. Re-ranking runs a **cross-encoder model** over the query and candidate passages jointly (`Query + Passage -> Relevance Score`).

#### ⚡ Current Implementation: FlashRank
* **Default Model:** `ms-marco-MiniLM-L-12-v2` configured via `settings.RERANK_MODEL`.
* **Execution:** Super-fast ONNX-quantized CPU cross-encoder.
* **Fallback Behavior:** If `flashrank` is not installed or fails at runtime, `rag_service.py` logs a warning and gracefully falls back to the RRF rank ordering without throwing errors.

#### 🔄 Supported Alternatives & Integration:
1. **Cohere Rerank API (Cloud alternative):**
   * Replace `FlashRank` in `rag_service.py` with `cohere.Client().rerank(model="rerank-english-v3.0", query=query, documents=passages)`.
2. **SentenceTransformers CrossEncoder (Local alternative):**
   * Use `from sentence_transformers import CrossEncoder` with model `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

### 4. System Configuration Matrix & Impact Analysis ([`config.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/app/core/config.py))

Modifying values in `app/core/config.py` (or setting environment variables) changes system behavior dynamically:

| Config Variable | Default | Impact & System Trade-Offs |
| :--- | :--- | :--- |
| `HYBRID_FUSION_WEIGHT` | `0.5` | Ratio between Dense ($0.5$) and Sparse ($0.5$). Increasing to `0.8` prioritizes semantic meaning; decreasing to `0.2` favors exact keyword hits. |
| `HYBRID_DENSE_CANDIDATE_K` | `20` | Number of dense candidates fetched from Qdrant before fusion. Higher values improve recall for deep queries but increase Qdrant latency. |
| `HYBRID_SPARSE_CANDIDATE_K` | `20` | Number of sparse BM25 candidates fetched from Qdrant. Increasing improves keyword coverage. |
| `HYBRID_RRF_CANDIDATE_K` | `10` | Top merged candidates passed from RRF fusion to FlashRank. Higher values give FlashRank more context to re-order. |
| `HYBRID_ENABLE_RERANK` | `true` | Toggles cross-encoder re-ranking. Setting to `false` bypasses FlashRank for ultra-low latency requirements. |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank cross-encoder weights model. |
| `OLLAMA_MODEL` | `qwen3:0.6b` | LLM model for final RAG answer synthesis. |

---

### 5. Automated Empirical Evaluation Harness ([`run_eval.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/tests/run_eval.py) & [`eval.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/app/api/v1/endpoints/eval.py))

Phase 3 introduces an automated evaluation harness to quantitatively benchmark retrieval performance and generation accuracy across 3 execution strategies: `dense`, `hybrid`, and `hybrid_reranked`.

#### Evaluated Metrics & Formulas:
1. **Precision@K:** $\frac{|\text{Retrieved Chunks} \cap \text{Ground Truth Chunks}|}{K}$
2. **Recall@K:** $\frac{|\text{Retrieved Chunks} \cap \text{Ground Truth Chunks}|}{|\text{Ground Truth Chunks}|}$
3. **MRR (Mean Reciprocal Rank):** $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$
4. **Answer Correctness:** Cosine semantic similarity ($\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$) between LLM generated response and ground truth `expected_answer`.

#### Running Evaluation Benchmarks:
* **Via Python CLI:**
  ```bash
  python -m tests.run_eval
  ```
* **Via REST API:**
  ```http
  POST /api/v1/eval/run?k=3
  ```
* **Report Artifacts:** Saved automatically in [`docs/eval_reports/`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/docs/eval_reports/) as structured JSON and formatted Markdown tables.


---

## 🏗️ 2. Architectural Pipeline Flow

```
                               ┌────────────────────────────────┐
                               │ Document (PDF / DOCX / TXT)    │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Ingestion & Chunking Service     │
                              │ (Recursive / Fixed-Size Window)  │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Embedding Generation (FastEmbed) │
                              │ Dense (384-d) + Sparse (BM25)    │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Qdrant Vector Store Indexing     │
                              └────────────────┬─────────────────┘
                                               │
                                        Query Request
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Hybrid Retrieval (Dense + BM25)  │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Reciprocal Rank Fusion (RRF)     │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Re-Ranking Engine (FlashRank)    │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ LLM Synthesis (Ollama Llama 3.2) │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Answer + Sources + Latency Stats │
                              └──────────────────────────────────┘
```

---

## 📂 3. Directory Structure & Key Files

```
apexguard/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── ingest.py        # Ingestion API endpoints (Files & Raw Text)
│   │   │   ├── query.py         # Search & Synthesis API endpoints
│   │   │   ├── documents.py     # Vector document list/view endpoints
│   │   │   └── eval.py          # Phase 3 RAG & LLM Evaluation API endpoint
│   │   └── router.py            # API V1 Router setup
│   ├── core/
│   │   ├── config.py            # Pydantic environment configurations & defaults
│   │   └── logging.py           # Structured logger configuration
│   ├── interfaces/              # Abstract Base Classes for loose-coupling
│   ├── schemas/                 # Pydantic Request & Response Data Models
│   ├── services/
│   │   ├── embedding_service.py # FastEmbed local embedding model wrapper
│   │   ├── ingestion_service.py # Parsers, chunkers, and text extractors
│   │   ├── llm_service.py       # Ollama LLM integration service
│   │   ├── rag_service.py       # Retrieval, RRF Fusion, and FlashRank re-ranker
│   │   └── vector_service.py    # Qdrant client connection and payload indexing
│   └── main.py                  # FastAPI Application, Middleware & Web Dashboard UI
├── docs/
│   ├── eval_reports/            # Automated JSON & Markdown evaluation reports
│   └── PHASE_3_SPEC.md          # Phase 3 Empirical Evaluation Specification
├── tests/
│   ├── eval_dataset.json        # Benchmark dataset (questions, expected answers, ground truth)
│   ├── run_eval.py              # Phase 3 empirical evaluation execution harness
│   └── test_rag.py              # Automated pytest unit test suite
├── Dockerfile                   # Production Docker image blueprint
├── docker-compose.yml           # Multi-container orchestration (App + Qdrant)
├── requirements.txt             # Python dependencies list
└── README.md                    # Project Documentation & KT Guide
```

---

## 🛠️ 4. Tech Stack Reference

| Component | Technology | Default Configuration / Model | Purpose |
| :--- | :--- | :--- | :--- |
| **Language & Framework** | Python 3.11, FastAPI | `FastAPI v3.0.0` | High-performance async web framework |
| **Vector DB** | Qdrant | `localhost:6333` (Collection: `apexguard_rag`) | Vector store supporting payload metadata filtering and hybrid retrieval |
| **Dense Embeddings** | FastEmbed | `BAAI/bge-small-en-v1.5` (384-d) | ONNX local execution embedding pipeline |
| **Sparse Embeddings** | FastEmbed | `Qdrant/bm25` | Sparse vector term-frequency keyword index |
| **Re-ranker** | FlashRank | `ms-marco-MiniLM-L-12-v2` | Lightweight neural cross-encoder for score refinement |
| **Evaluation Harness** | Custom Engine | Precision@K, Recall@K, MRR, Semantic Similarity | Quantitative evaluation of retrieval and generation |
| **LLM Engine** | Ollama | `qwen3:0.6b` (via `http://localhost:11434`) | Local execution of open-weights LLMs |
| **Containerization** | Docker, Docker Compose | Multi-container setup | Isolated environment build and execution |

---

## 📌 5. Key API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Live system diagnostic check (Qdrant status, embedding status, Ollama status, Phase 3 online) |
| `POST` | `/api/v1/ingest/file` | Ingest PDF, DOCX, or TXT file into Qdrant vector store |
| `POST` | `/api/v1/ingest/text` | Ingest raw text string directly into Qdrant vector store |
| `POST` | `/api/v1/query` | Execute RAG pipeline (Retrieval + RRF Fusion + FlashRank + LLM generation) |
| `GET` | `/api/v1/documents` | Inspect and filter stored vector chunks in Qdrant |
| `POST` | `/api/v1/eval/run` | Execute Phase 3 automated empirical evaluation harness (`dense` vs `hybrid` vs `hybrid_reranked`) |


---

## 🚀 6. Getting Started (For Beginners)

### Option 1: Quickstart via Docker Compose (Recommended)

1. Ensure Docker Desktop is installed and running.
2. Start all services:
   ```bash
   docker-compose up -d
   ```
3. Access Web Dashboard: Open [http://localhost:8000](http://localhost:8000) in your browser.
4. Access Interactive API Documentation: Open [http://localhost:8000/docs](http://localhost:8000/docs).

### Option 2: Local Python Virtual Environment

1. **Create and Activate venv:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Ollama Engine (In a separate terminal):**
   ```bash
   ollama run qwen3:0.6b
   ```
4. **Launch ApexGuard Server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

## 🧪 7. Running Unit Tests

Run the full automated test suite using `python -m pytest`:
```bash
python -m pytest tests/
```


