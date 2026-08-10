from fastapi import APIRouter
from app.core.logging import get_logger
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
