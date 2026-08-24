import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from app.core.config import settings
from app.core.logging import get_logger
from app.api.v1.router import api_router
from app.services.vector_service import vector_service

logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="ApexGuard Phase 4: Production LLM Gateway Core",
    version="3.0.0"
)

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Register API routers
app.include_router(api_router, prefix=settings.API_V1_STR)

from app.core.middleware import setup_middlewares
setup_middlewares(app)


# ─── Startup Event ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_banner():
    logger.info("=" * 60)
    logger.info("🛡️  ApexGuard RAG Gateway — Phase 4 (Production Gateway Core)")
    logger.info("=" * 60)
    logger.info(f"Project       : {settings.PROJECT_NAME}")
    logger.info(f"API Prefix    : {settings.API_V1_STR}")
    logger.info(f"Embedding     : {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIMENSION})")
    logger.info(f"Qdrant        : {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    logger.info(f"Qdrant Status : {'✅ Connected (persistent)' if vector_service.is_persistent else '⚠️  In-Memory (non-persistent)'}")
    logger.info(f"Ollama        : {settings.OLLAMA_BASE_URL} (model: {settings.OLLAMA_MODEL})")
    logger.info(f"Upload Dir    : {settings.UPLOAD_DIR}")
    logger.info("=" * 60)
    logger.info(f"Swagger UI    : http://localhost:8000/docs")
    logger.info(f"Dashboard     : http://localhost:8000/")
    logger.info("=" * 60)


# ─── Health Check ───────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "system": "ApexGuard RAG Gateway",
        "phase": 4,
        "embedding_model": settings.EMBEDDING_MODEL,
        "qdrant_host": settings.QDRANT_HOST,
        "qdrant_persistent": vector_service.is_persistent,
        "ollama_url": settings.OLLAMA_BASE_URL
    }



# ─── Dashboard UI ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ApexGuard RAG Gateway</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0d1117;
                --card-bg: #161b22;
                --border: #30363d;
                --accent: #58a6ff;
                --accent-hover: #1f6feb;
                --text: #c9d1d9;
                --text-muted: #8b949e;
                --success: #238636;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 40px 20px;
                display: flex;
                justify-content: center;
            }
            .container {
                max-width: 900px;
                width: 100%;
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                color: #ffffff;
            }
            .header p {
                color: var(--text-muted);
                font-size: 1.1rem;
            }
            .card {
                background-color: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 24px;
            }
            .card h2 {
                margin-top: 0;
                font-size: 1.3rem;
                color: var(--accent);
            }
            .form-group {
                margin-bottom: 16px;
            }
            .form-row {
                display: flex;
                gap: 16px;
            }
            .form-row .form-group {
                flex: 1;
            }
            .mode-row {
                display: flex;
                gap: 8px;
                margin-bottom: 16px;
            }
            .mode-option {
                flex: 1;
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 10px;
                background: #0d1117;
                border: 1px solid var(--border);
                border-radius: 6px;
                cursor: pointer;
            }
            .mode-option input {
                width: auto;
                margin: 0;
            }
            .advanced-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 16px;
            }
            .toggle-row {
                display: flex;
                align-items: center;
                gap: 8px;
                min-height: 38px;
            }
            .hidden {
                display: none;
            }
            @media (max-width: 720px) {
                .form-row,
                .mode-row,
                .advanced-grid {
                    display: block;
                }
            }
            label {
                display: block;
                margin-bottom: 6px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            input[type="text"], input[type="number"], input[type="file"], textarea, select {
                width: 100%;
                padding: 10px;
                background: #0d1117;
                border: 1px solid var(--border);
                border-radius: 6px;
                color: var(--text);
                font-family: inherit;
                box-sizing: border-box;
            }
            button {
                background-color: var(--accent);
                color: #ffffff;
                border: none;
                padding: 10px 18px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: background 0.2s;
            }
            button:hover {
                background-color: var(--accent-hover);
            }
            .response-box {
                background-color: #0d1117;
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 12px;
                white-space: pre-wrap;
                font-family: monospace;
                font-size: 0.85rem;
                color: #e6edf3;
                max-height: 250px;
                overflow-y: auto;
                margin-top: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ ApexGuard RAG Gateway</h1>
                <p>Phase 2 Hybrid RAG Engine — dense retrieval, sparse BM25 search, RRF fusion, and optional re-ranking.</p>
            </div>

            <div class="card">
                <h2>1. Ingest Document File</h2>
                <form id="uploadForm">
                    <div class="form-group">
                        <label>Select File (PDF, DOCX, TXT)</label>
                        <input type="file" id="fileInput" required />
                    </div>

                    <!-- Added Chunking Options UI -->
                    <div class="form-row">
                        <div class="form-group">
                            <label>Chunking Strategy</label>
                            <select id="strategyInput">
                                <option value="semantic" selected>Semantic Chunking</option>
                                <option value="recursive">Recursive Character</option>
                                <option value="sentence">Sentence Boundary</option>
                                <option value="fixed">Fixed Size</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Chunk Size</label>
                            <input type="number" id="chunkSizeInput" value="500" />
                        </div>
                        <div class="form-group">
                            <label>Chunk Overlap</label>
                            <input type="number" id="chunkOverlapInput" value="50" />
                        </div>
                    </div>

                    <button type="submit">Ingest File</button>
                </form>
                <div id="uploadResult" class="response-box" style="display:none;"></div>
            </div>

            <div class="card">
                <h2>2. Query RAG Engine</h2>
                <form id="queryForm">
                    <div class="mode-row">
                        <label class="mode-option">
                            <input type="radio" name="queryMode" value="dense" checked />
                            Dense Query
                        </label>
                        <label class="mode-option">
                            <input type="radio" name="queryMode" value="hybrid" />
                            Sparse Hybrid Query
                        </label>
                    </div>
                    <div class="form-group">
                        <label>Your Question</label>
                        <input type="text" id="queryInput" placeholder="Ask something about ingested docs..." required />
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Relevant Items</label>
                            <input type="number" id="topKInput" value="3" min="1" max="20" />
                        </div>
                        <div class="form-group">
                            <label>Filename Filter</label>
                            <input type="text" id="filenameFilterInput" placeholder="Optional exact filename" />
                        </div>
                    </div>
                    <div id="hybridControls" class="hidden">
                        <div class="advanced-grid">
                            <div class="form-group">
                                <label>Dense Weight</label>
                                <input type="number" id="fusionWeightInput" value="0.5" min="0" max="1" step="0.05" />
                            </div>
                            <div class="form-group">
                                <label>RRF Candidates</label>
                                <input type="number" id="rrfCandidateKInput" value="10" min="1" max="100" />
                            </div>
                            <div class="form-group">
                                <label>Dense Candidates</label>
                                <input type="number" id="denseCandidateKInput" value="20" min="1" max="100" />
                            </div>
                            <div class="form-group">
                                <label>Sparse Candidates</label>
                                <input type="number" id="sparseCandidateKInput" value="20" min="1" max="100" />
                            </div>
                            <div class="form-group">
                                <label>Re-rank</label>
                                <div class="toggle-row">
                                    <input type="checkbox" id="enableRerankInput" checked />
                                    <span>FlashRank</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <button type="submit">Search & Synthesize</button>
                </form>
                <div id="queryResult" class="response-box" style="display:none;"></div>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const strategyInput = document.getElementById('strategyInput').value;
                const chunkSizeInput = document.getElementById('chunkSizeInput').value;
                const chunkOverlapInput = document.getElementById('chunkOverlapInput').value;

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('chunk_strategy', strategyInput);
                formData.append('chunk_size', chunkSizeInput);
                formData.append('chunk_overlap', chunkOverlapInput);

                const resBox = document.getElementById('uploadResult');
                resBox.style.display = 'block';
                resBox.innerText = `Ingesting with [${strategyInput}] chunking...`;

                try {
                    const res = await fetch('/api/v1/ingest/file', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    resBox.innerText = JSON.stringify(data, null, 2);
                } catch (err) {
                    resBox.innerText = 'Error: ' + err.message;
                }
            });

            document.querySelectorAll('input[name="queryMode"]').forEach((input) => {
                input.addEventListener('change', () => {
                    const mode = document.querySelector('input[name="queryMode"]:checked').value;
                    document.getElementById('hybridControls').classList.toggle('hidden', mode !== 'hybrid');
                });
            });

            document.getElementById('queryForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = document.getElementById('queryInput').value;
                const mode = document.querySelector('input[name="queryMode"]:checked').value;
                const topK = Number(document.getElementById('topKInput').value || 3);
                const filenameFilter = document.getElementById('filenameFilterInput').value.trim();
                const resBox = document.getElementById('queryResult');
                resBox.style.display = 'block';
                resBox.innerText = mode === 'hybrid'
                    ? 'Running dense + sparse retrieval, fusion, and synthesis...'
                    : 'Running dense retrieval and synthesis...';

                const requestBody = {
                    query: query,
                    top_k: topK
                };
                if (filenameFilter) {
                    requestBody.filename_filter = filenameFilter;
                }

                let endpoint = '/api/v1/query';
                if (mode === 'hybrid') {
                    endpoint = '/api/v1/query/hybrid';
                    requestBody.fusion_weight = Number(document.getElementById('fusionWeightInput').value || 0.5);
                    requestBody.enable_rerank = document.getElementById('enableRerankInput').checked;
                    requestBody.rrf_candidate_k = Number(document.getElementById('rrfCandidateKInput').value || 10);
                    requestBody.dense_candidate_k = Number(document.getElementById('denseCandidateKInput').value || 20);
                    requestBody.sparse_candidate_k = Number(document.getElementById('sparseCandidateKInput').value || 20);
                }

                try {
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });
                    const data = await res.json();
                    resBox.innerText = JSON.stringify(data, null, 2);
                } catch (err) {
                    resBox.innerText = 'Error: ' + err.message;
                }
            });
        </script>
    </body>
    </html>
    """
