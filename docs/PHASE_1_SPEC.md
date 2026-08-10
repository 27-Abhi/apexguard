# ApexGuard Phase 1 — Technical Specification & Verification

## Overview
Phase 1 delivers the fundamental building blocks of the ApexGuard RAG pipeline. It handles document parsing, hierarchical chunking, vector embedding, similarity search in Qdrant, and context-augmented generation via Ollama.

## Architecture

```
[Uploaded Document] -> [Ingestion Parser] -> [Chunker (Recursive/Fixed)] -> [FastEmbed ONNX]
                                                                                  │
                                                                                  ▼
[User Query] --------------> [Retriever] <---------------------------------- [Qdrant DB]
                                  │
                                  ▼
                        [Formatted Context] -> [Ollama LLM] -> [Answer + Sources]
```

## Module Specifications

1. **[`api/ingestion.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/api/ingestion.py):**
   - Supports PDF (`pypdf`), DOCX (`docx2txt`), and plain text format extraction.
   - Provides `recursive` character splitting with customizable chunk size and overlap, alongside `fixed` window splitting.

2. **[`api/embeddings.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/api/embeddings.py):**
   - Utilizes `FastEmbed` with `BAAI/bge-small-en-v1.5` (384 dimensions).
   - Generates local ONNX embeddings without external paid API calls.

3. **[`api/vector_store.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/api/vector_store.py):**
   - Integrates with Qdrant vector database using Cosine distance.
   - Features payload indexing, UUID generation, metadata filtering, and automatic fallback to memory mode when standalone.

4. **[`api/retriever.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/api/retriever.py):**
   - Fetches Top-K relevant chunks, converts distance scores, and formats structured prompts with document provenance.

5. **[`api/llm.py`](file:///C:/Users/abhinav.kuppasad/Downloads/apexguard/api/llm.py):**
   - Asynchronous HTTP client targeting Ollama endpoints (`http://localhost:11434`). Supports streaming and standalone fallback.

## Endpoints

- `GET /health` - System health and component connection metrics.
- `POST /api/v1/ingest/file` - Multipart document upload & vectorization.
- `POST /api/v1/ingest/text` - Direct raw text snippet vectorization.
- `POST /api/v1/query` - Vector similarity retrieval + answer generation.
- `GET /api/v1/documents` - Vector store payload inspector.
- `GET /` - Visual dashboard control center.
