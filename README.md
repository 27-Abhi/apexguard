# ApexGuard — Phase 1 RAG Engine

ApexGuard is a high-performance, real-time LLM Gateway and AI Agent Platform. **Phase 1** focuses on building a rock-solid, production-grade local Retrieval-Augmented Generation (RAG) pipeline.

---

## 🏗️ Architecture & Pipeline Flow

```
Documents (PDF / DOCX / TXT)
   │
   ▼
Ingestion (pypdf / docx2txt)
   │
   ▼
Chunking Strategies (Recursive Character & Fixed Size Window)
   │
   ▼
Embedding Generation (FastEmbed / ONNX local embeddings)
   │
   ▼
Vector Storage & Indexing (Qdrant Cosine Similarity)
   │
   ▼
Top-K Retriever & Metadata Filtering
   │
   ▼
LLM Generation & Context Assembly (Ollama / Llama 3.2)
   │
   ▼
Structured Answer + Ranked Sources & Latency Telemetry
```

---

## 🛠️ Tech Stack & Dependencies

- **Language & Web Framework:** Python 3.11, [FastAPI](https://fastapi.tiangolo.com/)
- **Vector Database:** [Qdrant](https://qdrant.tech/)
- **Embedding Engine:** FastEmbed / Sentence Transformers (`BAAI/bge-small-en-v1.5`)
- **LLM Serving:** Ollama (`llama3.2` or configurable)
- **Containerization:** Docker & Docker Compose

---

## 🚀 Quickstart Guide

### Option 1: Docker Compose (Recommended)

1. **Launch Containers:**
   ```bash
   docker-compose up -d
   ```
2. **Access Web UI Control Center:**
   Open browser at [http://localhost:8000](http://localhost:8000)

3. **Access Interactive API Docs:**
   Open Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2: Local Python Virtual Environment

1. **Create and Activate venv:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start FastAPI Gateway:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

## 📌 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check pipeline components status |
| `POST` | `/api/v1/ingest/file` | Ingest PDF/DOCX/TXT file with chunking strategy |
| `POST` | `/api/v1/ingest/text` | Ingest raw text snippet directly |
| `POST` | `/api/v1/query` | Execute vector retrieval + LLM synthesis |
| `GET` | `/api/v1/documents` | List indexed vector chunks |

---

## 🧪 Testing

Run pytest suite:
```bash
pytest tests/
```
