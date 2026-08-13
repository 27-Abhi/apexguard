import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.logging import get_logger
from app.schemas.query import HybridQueryRequest, QueryRequest, RAGResponse, SourceResponse
from app.services.rag_service import retriever_service
from app.services.llm_service import llm_service

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["RAG Query Engine"])

@router.post("", response_model=RAGResponse)
async def query_rag_pipeline(payload: QueryRequest):
    """Execute vector retrieval + LLM synthesis."""
    start_time = time.time()
    logger.info(f"POST /api/v1/query — query='{payload.query[:100]}', top_k={payload.top_k}, stream={payload.stream}, filter={payload.filename_filter}")
    
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
        logger.info("Streaming response mode enabled")
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

    logger.info(f"Query pipeline complete: {len(sources)} sources, {elapsed_ms}ms total latency")

    return RAGResponse(
        query=payload.query,
        answer=answer,
        sources=sources,
        latency_ms=elapsed_ms
    )


@router.post("/hybrid", response_model=RAGResponse)
async def query_hybrid_rag_pipeline(payload: HybridQueryRequest):
    """Execute dense + sparse hybrid retrieval with RRF and optional re-ranking."""
    start_time = time.time()
    logger.info(f"POST /api/v1/query/hybrid - query='{payload.query[:100]}', top_k={payload.top_k}, rerank={payload.enable_rerank}")

    metadata_filter = {"filename": payload.filename_filter} if payload.filename_filter else None
    search_results = retriever_service.retrieve_hybrid(
        query=payload.query,
        top_k=payload.top_k,
        metadata_filter=metadata_filter,
        fusion_weight=payload.fusion_weight,
        enable_rerank=payload.enable_rerank,
        rrf_candidate_k=payload.rrf_candidate_k,
        dense_candidate_k=payload.dense_candidate_k,
        sparse_candidate_k=payload.sparse_candidate_k
    )

    context_str = retriever_service.format_context(search_results)
    answer = await llm_service.generate_answer(payload.query, context_str)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return RAGResponse(
        query=payload.query,
        answer=answer,
        sources=[
            SourceResponse(text=res["text"], score=res["score"], metadata=res["metadata"])
            for res in search_results
        ],
        latency_ms=elapsed_ms
    )
