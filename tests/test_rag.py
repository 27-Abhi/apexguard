import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["phase"] == 3

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

def test_query_hybrid_rag():
    query_payload = {
        "query": "What is ApexGuard?",
        "top_k": 2,
        "rrf_candidate_k": 5,
        "dense_candidate_k": 5,
        "sparse_candidate_k": 5,
        "enable_rerank": False
    }
    response = client.post("/api/v1/query/hybrid", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) <= 2

def test_eval_endpoint():
    response = client.post("/api/v1/eval/run?k=2")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["benchmark_questions"] >= 1
    assert data["questions_evaluated"] <= 5
    assert data["strategies_evaluated"] == ["dense"]
    assert data["include_generation"] is False
    assert "results" in data
    assert len(data["results"]) == 1

