import os
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.ingest import TextIngestRequest, IngestResponse
from app.services.ingestion_service import DocumentIngestionService, DocumentChunkerService
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
):
    """Ingest PDF, DOCX, or text file into the RAG pipeline."""
    start_time = time.time()
    logger.info(f"POST /api/v1/ingest/file — file='{file.filename}', strategy={chunk_strategy}, chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    logger.info(f"File saved to disk: {file_path} ({len(content)} bytes)")
        
    extracted_text = DocumentIngestionService.extract_text_from_file(file_path, file.filename)
    if not extracted_text.strip():
        logger.error(f"No readable text extracted from '{file.filename}'")
        raise HTTPException(status_code=400, detail="No readable text extracted from file.")

    chunks = DocumentChunkerService.chunk_document(
        text=extracted_text, 
        strategy=chunk_strategy, 
        chunk_size=chunk_size, 
        overlap=chunk_overlap
    )

    embeddings = embedding_service.embed_texts(chunks)
    metadatas = [
        {
            "filename": file.filename,
            "chunk_index": idx,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks)
        }
        for idx in range(len(chunks))
    ]

    point_ids = vector_service.insert_documents(chunks, embeddings, metadatas)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(f"File ingestion complete: '{file.filename}' → {len(chunks)} chunks in {elapsed_ms}ms")

    return IngestResponse(
        status="success",
        filename=file.filename,
        chunks_created=len(chunks),
        point_ids=point_ids,
        latency_ms=elapsed_ms
    )

@router.post("/text", response_model=IngestResponse)
def ingest_raw_text(payload: TextIngestRequest):
    """Ingest plain text snippet directly."""
    start_time = time.time()
    logger.info(f"POST /api/v1/ingest/text — title='{payload.title}', strategy={payload.chunk_strategy}, content_length={len(payload.content)}")
    
    if not payload.content.strip():
        logger.error("Empty content received for text ingestion")
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    filename = f"text_{payload.title.replace(' ', '_').lower()}.txt"
    chunks = DocumentChunkerService.chunk_document(
        text=payload.content,
        strategy=payload.chunk_strategy,
        chunk_size=payload.chunk_size,
        overlap=payload.chunk_overlap
    )

    embeddings = embedding_service.embed_texts(chunks)
    metadatas = [
        {
            "filename": filename,
            "title": payload.title,
            "chunk_index": idx,
            "chunk_strategy": payload.chunk_strategy,
            "chunk_size": payload.chunk_size,
            "chunk_overlap": payload.chunk_overlap,
            "total_chunks": len(chunks)
        }
        for idx in range(len(chunks))
    ]

    point_ids = vector_service.insert_documents(chunks, embeddings, metadatas)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(f"Text ingestion complete: '{payload.title}' → {len(chunks)} chunks in {elapsed_ms}ms")

    return IngestResponse(
        status="success",
        filename=filename,
        chunks_created=len(chunks),
        point_ids=point_ids,
        latency_ms=elapsed_ms
    )
