from fastapi import APIRouter
from app.services.vector_service import vector_service

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.get("")
def list_indexed_documents():
    """List indexed document chunks in vector storage."""
    docs = vector_service.get_all_documents()
    return {
        "total_chunks": len(docs),
        "documents": docs
    }
