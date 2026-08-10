import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.core.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="ApexGuard Phase 1: High-Performance Real-Time RAG Engine",
    version="1.0.0"
)

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Register API routers
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "system": "ApexGuard RAG Gateway",
        "phase": 1,
        "embedding_model": settings.EMBEDDING_MODEL,
        "qdrant_host": settings.QDRANT_HOST,
        "ollama_url": settings.OLLAMA_BASE_URL
    }

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
            label {
                display: block;
                margin-bottom: 6px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            input[type="text"], textarea, select {
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
                <p>Phase 1 Production RAG Engine — local embeddings, Qdrant vector storage, and Ollama integration.</p>
            </div>

            <div class="card">
                <h2>1. Ingest Document File</h2>
                <form id="uploadForm">
                    <div class="form-group">
                        <label>Select File (PDF, DOCX, TXT)</label>
                        <input type="file" id="fileInput" required />
                    </div>
                    <button type="submit">Ingest File</button>
                </form>
                <div id="uploadResult" class="response-box" style="display:none;"></div>
            </div>

            <div class="card">
                <h2>2. Query RAG Engine</h2>
                <form id="queryForm">
                    <div class="form-group">
                        <label>Your Question</label>
                        <input type="text" id="queryInput" placeholder="Ask something about ingested docs..." required />
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
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const resBox = document.getElementById('uploadResult');
                resBox.style.display = 'block';
                resBox.innerText = 'Ingesting and embedding...';

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

            document.getElementById('queryForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = document.getElementById('queryInput').value;
                const resBox = document.getElementById('queryResult');
                resBox.style.display = 'block';
                resBox.innerText = 'Searching vector store and calling LLM...';

                try {
                    const res = await fetch('/api/v1/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query, top_k: 3 })
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
