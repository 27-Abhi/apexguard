import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["phase"] == 1

def test_ingest_text():
    payload = {
        "title": "ApexGuard Architecture Test",
        "content": "ApexGuard is a Real-Time LLM Gateway designed for low-latency AI agent routing and vector retrieval.",
        "chunk_strategy": "recursive",
        "chunk_size": 200,
        "chunk_overlap": 20
    }
    response = client.post("/api/v1/ingest/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] >= 1

def test_query_rag():
    query_payload = {
        "query": "What is ApexGuard?",
        "top_k": 3
    }
    response = client.post("/api/v1/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
