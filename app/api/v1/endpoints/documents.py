from typing import Optional, List

from fastapi import APIRouter, HTTPException
from app.core.logging import get_logger
from app.schemas.ingest import DeleteRequest, DeleteResponse
from app.services.vector_service import vector_service

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.get("")
def list_indexed_documents():
    """List indexed document chunks in vector storage."""
    logger.info("GET /api/v1/documents — listing indexed chunks")
    docs = vector_service.get_all_documents()
    logger.info(f"Returning {len(docs)} document chunks")
    return {
        "total_chunks": len(docs),
        "documents": docs
    }

@router.delete("", response_model=DeleteResponse)
def delete_indexed_documents(payload: DeleteRequest):
    """Delete indexed points by filename or by point ids."""
    if not payload.filename and not payload.point_ids:
        logger.error("DELETE /api/v1/documents — missing filename and point_ids")
        raise HTTPException(status_code=400, detail="Provide filename or point_ids to delete documents.")

    metadata_filter = {"filename": payload.filename} if payload.filename else None
    deleted = vector_service.delete_documents(
        point_ids=payload.point_ids,
        metadata_filter=metadata_filter
    )

    if not deleted:
        logger.warning("DELETE /api/v1/documents — no documents deleted")

    return DeleteResponse(status="success", deleted=deleted)
