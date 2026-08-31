from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from app.schemas.router import (
    RouteDecision,
    RouterClassifyRequest,
    SmartQueryRequest,
    SmartQueryResponse,
    RouteType,
)
from app.services.router_service import router_service
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/router", tags=["Intelligent Model Router"])


@router.post("/classify", response_model=RouteDecision)
def classify_query(body: RouterClassifyRequest):
    """
    Sub-millisecond query intent & complexity classifier.
    Determines optimal route (DIRECT_FAST vs KNOWLEDGE_RAG vs COMPLEX_REASONING)
    without running inference.
    """
    try:
        decision = router_service.classify(body.query, override_route=body.override_route)
        return decision
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=SmartQueryResponse)
async def smart_query(body: SmartQueryRequest):
    """
    Execute intelligent query routing.
    Dynamically routes to small fast model, RAG pipeline, or reasoning model
    based on query complexity and intent.
    """
    try:
        response = await router_service.execute_route(body)
        return response
    except Exception as e:
        logger.error(f"Smart query routing execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/routes", response_model=Dict[str, Any])
def list_routes():
    """
    List configured routing strategies, model targets, and latency/cost trade-offs.
    """
    return {
        "active_strategy": settings.ROUTER_DEFAULT_STRATEGY,
        "models": {
            "fast": settings.ROUTER_FAST_MODEL,
            "reasoning": settings.ROUTER_REASONING_MODEL,
            "rag": settings.OLLAMA_MODEL
        },
        "routes": [
            {
                "route": RouteType.DIRECT_FAST.value,
                "target_model": settings.ROUTER_FAST_MODEL,
                "complexity": "simple",
                "estimated_latency": "< 150ms",
                "estimated_cost": "free / local",
                "description": "Direct fast LLM inference for greetings, chitchat, and simple tasks (0 retrieval cost)"
            },
            {
                "route": RouteType.KNOWLEDGE_RAG.value,
                "target_model": settings.OLLAMA_MODEL,
                "complexity": "moderate",
                "estimated_latency": "300ms - 800ms",
                "estimated_cost": "low",
                "description": "Hybrid vector/sparse retrieval with FlashRank re-ranking + grounded synthesis"
            },
            {
                "route": RouteType.COMPLEX_REASONING.value,
                "target_model": settings.ROUTER_REASONING_MODEL,
                "complexity": "complex",
                "estimated_latency": "1s - 4s",
                "estimated_cost": "medium",
                "description": "High-parameter reasoning model for coding, mathematics, and multi-step logic"
            }
        ]
    }
