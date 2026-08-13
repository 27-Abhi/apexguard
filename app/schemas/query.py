from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.core.config import settings

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(5, description="Number of top context results to retrieve")
    filename_filter: Optional[str] = Field(None, description="Optional metadata filter by filename")
    stream: bool = Field(False, description="Whether to stream the response tokens")

class HybridQueryRequest(QueryRequest):
    top_k: int = Field(settings.HYBRID_TOP_K, description="Final number of relevant context results to return")
    fusion_weight: float = Field(settings.HYBRID_FUSION_WEIGHT, ge=0.0, le=1.0, description="Dense search weight; sparse uses 1 - weight")
    enable_rerank: bool = Field(settings.HYBRID_ENABLE_RERANK, description="Whether to run FlashRank re-ranking")
    rrf_candidate_k: int = Field(settings.HYBRID_RRF_CANDIDATE_K, description="Number of RRF candidates to pass to reranker")
    dense_candidate_k: int = Field(settings.HYBRID_DENSE_CANDIDATE_K, description="Dense candidate count before fusion")
    sparse_candidate_k: int = Field(settings.HYBRID_SPARSE_CANDIDATE_K, description="Sparse candidate count before fusion")

class SourceResponse(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceResponse]
    latency_ms: float
