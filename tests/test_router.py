import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.schemas.router import RouteType, ComplexityLevel
from app.services.router_service import router_service

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {settings.GATEWAY_API_KEY}"}


def test_router_classify_greetings():
    decision = router_service.classify("Hello there! Good morning.")
    assert decision.route == RouteType.DIRECT_FAST
    assert decision.complexity == ComplexityLevel.SIMPLE
    assert decision.requires_rag is False
    assert decision.model == settings.ROUTER_FAST_MODEL
    assert decision.classification_time_ms < 50.0


def test_router_classify_general_everyday_queries():
    queries = [
        "WHAT IS THE TIME",
        "What is the date today?",
        "Tell me a joke",
        "What is 2 + 2?",
        "What is the capital of France?"
    ]
    for q in queries:
        decision = router_service.classify(q)
        assert decision.route == RouteType.DIRECT_FAST, f"Query '{q}' failed: expected direct_fast, got {decision.route}"
        assert decision.requires_rag is False
        assert decision.complexity == ComplexityLevel.SIMPLE



def test_router_classify_reasoning():
    decision = router_service.classify(
        "Write a python function to compute the Fibonacci sequence using dynamic programming and analyze the big o time complexity step by step."
    )
    assert decision.route == RouteType.COMPLEX_REASONING
    assert decision.complexity == ComplexityLevel.COMPLEX
    assert decision.requires_rag is False
    assert decision.model == settings.ROUTER_REASONING_MODEL


def test_router_classify_knowledge_rag():
    decision = router_service.classify(
        "What is ApexGuard and how is hybrid retrieval configured with Qdrant?"
    )
    assert decision.route == RouteType.KNOWLEDGE_RAG
    assert decision.complexity == ComplexityLevel.MODERATE
    assert decision.requires_rag is True
    assert decision.model == settings.OLLAMA_MODEL


def test_router_classify_override():
    decision = router_service.classify("Hi", override_route=RouteType.COMPLEX_REASONING)
    assert decision.route == RouteType.COMPLEX_REASONING
    assert decision.confidence == 1.0


def test_api_router_classify_endpoint():
    payload = {"query": "What is the Qdrant collection name in settings?"}
    response = client.post("/api/v1/router/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "knowledge_rag"
    assert data["requires_rag"] is True
    assert "classification_time_ms" in data


def test_api_router_routes_list_endpoint():
    response = client.get("/api/v1/router/routes")
    assert response.status_code == 200
    data = response.json()
    assert "routes" in data
    assert len(data["routes"]) == 3
    assert "models" in data


def test_api_router_smart_query():
    payload = {
        "query": "Hello, how are you?",
        "override_route": "direct_fast"
    }
    response = client.post("/api/v1/router/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["routing"]["route"] == "direct_fast"
    assert "latency_ms" in data


def test_gateway_auto_model_routing():
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
    response = client.post("/api/v1/gateway/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "routing" in data
    assert data["routing"]["route"] == "direct_fast"
    assert data["model"] == settings.ROUTER_FAST_MODEL
