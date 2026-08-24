import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ApexGuard - Phase 2 RAG Platform"
    API_V1_STR: str = "/api/v1"
    
    # Qdrant settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "apexguard_rag")
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION: int = 384
    SPARSE_EMBEDDING_MODEL: str = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
    SPARSE_VECTOR_NAME: str = os.getenv("SPARSE_VECTOR_NAME", "text-sparse")
    
    # Hybrid retrieval settings
    HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))
    HYBRID_RELEVANT_ITEMS: int = int(os.getenv("HYBRID_RELEVANT_ITEMS", "3"))
    HYBRID_TOP_K: int = int(os.getenv("HYB  RID_TOP_K", os.getenv("HYBRID_RELEVANT_ITEMS", "3")))
    HYBRID_RRF_CANDIDATE_K: int = int(os.getenv("HYBRID_RRF_CANDIDATE_K", "15"))
    HYBRID_DENSE_CANDIDATE_K: int = int(os.getenv("HYBRID_DENSE_CANDIDATE_K", "20"))
    HYBRID_SPARSE_CANDIDATE_K: int = int(os.getenv("HYBRID_SPARSE_CANDIDATE_K", "20"))
    HYBRID_FUSION_WEIGHT: float = float(os.getenv("HYBRID_FUSION_WEIGHT", "0.5"))
    HYBRID_ENABLE_RERANK: bool = os.getenv("HYBRID_ENABLE_RERANK", "true").lower() in ("1", "true", "yes", "on")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
    
    # Ollama LLM settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    
    # Upload storage settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")

    # Gateway settings (Phase 4)
    GATEWAY_API_KEY: str = os.getenv("GATEWAY_API_KEY", "sk-apexguard-dev-key")
    GATEWAY_TIMEOUT_SEC: float = float(os.getenv("GATEWAY_TIMEOUT_SEC", "30.0"))
    GATEWAY_MAX_RETRIES: int = int(os.getenv("GATEWAY_MAX_RETRIES", "3"))

    class Config:
        case_sensitive = True

settings = Settings()
