from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(5, description="Number of top context results to retrieve")
    filename_filter: Optional[str] = Field(None, description="Optional metadata filter by filename")
    stream: bool = Field(False, description="Whether to stream the response tokens")

class SourceResponse(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceResponse]
    latency_ms: float
