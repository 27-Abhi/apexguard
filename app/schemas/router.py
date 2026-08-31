from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.schemas.query import SourceResponse


class RouteType(str, Enum):
    DIRECT_FAST = "direct_fast"
    KNOWLEDGE_RAG = "knowledge_rag"
    COMPLEX_REASONING = "complex_reasoning"


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class RouteDecision(BaseModel):
    query: str
    route: RouteType
    model: str
    complexity: ComplexityLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    estimated_latency_tier: str
    estimated_cost_tier: str
    requires_rag: bool
    classification_time_ms: float


class RouterClassifyRequest(BaseModel):
    query: str = Field(..., description="The user prompt or query to classify and route")
    override_route: Optional[RouteType] = Field(None, description="Optional manual override route type")


class SmartQueryRequest(BaseModel):
    query: str = Field(..., description="The user query to be dynamically routed and executed")
    override_route: Optional[RouteType] = Field(None, description="Force a specific route instead of automated classification")
    top_k: Optional[int] = Field(3, description="Number of context documents if routed to RAG")
    stream: bool = Field(False, description="Whether to stream the LLM response tokens")
    temperature: Optional[float] = Field(0.7, description="Generation temperature")


class SmartQueryResponse(BaseModel):
    query: str
    answer: str
    routing: RouteDecision
    sources: List[SourceResponse] = Field(default_factory=list)
    latency_ms: float
