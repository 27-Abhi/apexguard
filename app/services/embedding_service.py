from typing import List
from fastembed import SparseTextEmbedding, TextEmbedding
from app.core.config import settings
from app.core.logging import get_logger
from app.interfaces.embeddings import BaseEmbeddingService

logger = get_logger(__name__)

class FastEmbedService(BaseEmbeddingService):
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        logger.info(f"Embedding service initialized (model: {self.model_name}, dim: {settings.EMBEDDING_DIMENSION})")

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' (first-time download if not cached)...")
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info(f"Embedding model '{self.model_name}' loaded successfully")
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("embed_texts called with empty input")
            return []
        logger.info(f"Embedding {len(texts)} text chunks...")
        embeddings = list(self.model.embed(texts))
        result = [e.tolist() for e in embeddings]
        logger.info(f"Generated {len(result)} embeddings (dim={len(result[0])})")
        return result

    def embed_query(self, query: str) -> List[float]:
        logger.debug(f"Embedding query: '{query[:80]}...'") if len(query) > 80 else logger.debug(f"Embedding query: '{query}'")
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

embedding_service = FastEmbedService()


class FastEmbedSparseService:
    def __init__(self, model_name: str = settings.SPARSE_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        logger.info(f"Sparse embedding service initialized (model: {self.model_name})")

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading sparse embedding model '{self.model_name}'...")
            self._model = SparseTextEmbedding(model_name=self.model_name)
            logger.info(f"Sparse embedding model '{self.model_name}' loaded successfully")
        return self._model

    def embed_texts(self, texts: List[str]):
        if not texts:
            logger.warning("sparse embed_texts called with empty input")
            return []
        return list(self.model.embed(texts))

    def embed_query(self, query: str):
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else None


sparse_embedding_service = FastEmbedSparseService()
