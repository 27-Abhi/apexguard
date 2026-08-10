import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ApexGuard - Phase 1 RAG Platform"
    API_V1_STR: str = "/api/v1"
    
    # Qdrant settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "apexguard_rag")
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION: int = 384
    
    # Ollama LLM settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    
    # Upload storage settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")

    class Config:
        case_sensitive = True

settings = Settings()
