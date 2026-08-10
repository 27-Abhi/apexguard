from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TextIngestRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Raw text content to ingest")
    chunk_strategy: str = Field("recursive", description="Chunking strategy: 'recursive' or 'fixed'")
    chunk_size: int = Field(500, description="Size of chunk in characters")
    chunk_overlap: int = Field(50, description="Overlap between chunks in characters")

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    point_ids: List[str]
    latency_ms: float
