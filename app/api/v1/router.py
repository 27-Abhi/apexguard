from fastapi import APIRouter
from app.api.v1.endpoints import ingest, query, documents, eval

api_router = APIRouter()
api_router.include_router(ingest.router)
api_router.include_router(query.router)
api_router.include_router(documents.router)
api_router.include_router(eval.router)

