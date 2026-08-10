import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.query import QueryRequest, RAGResponse, SourceResponse
from app.services.rag_service import retriever_service
from app.services.llm_service import llm_service

router = APIRouter(prefix="/query", tags=["RAG Query Engine"])

@router.post("", response_model=RAGResponse)
async def query_rag_pipeline(payload: QueryRequest):
    """Execute vector retrieval + LLM synthesis."""
    start_time = time.time()
    
    metadata_filter = None
    if payload.filename_filter:
        metadata_filter = {"filename": payload.filename_filter}

    search_results = retriever_service.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        metadata_filter=metadata_filter
    )

    context_str = retriever_service.format_context(search_results)

    if payload.stream:
        return StreamingResponse(
            llm_service.stream_answer(payload.query, context_str),
            media_type="text/plain"
        )

    answer = await llm_service.generate_answer(payload.query, context_str)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    sources = [
        SourceResponse(
            text=res["text"],
            score=res["score"],
            metadata=res["metadata"]
        )
        for res in search_results
    ]

    return RAGResponse(
        query=payload.query,
        answer=answer,
        sources=sources,
        latency_ms=elapsed_ms
    )
