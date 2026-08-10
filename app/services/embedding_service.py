from typing import List
from fastembed import TextEmbedding
from app.core.config import settings
from app.interfaces.embeddings import BaseEmbeddingService

class FastEmbedService(BaseEmbeddingService):
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> List[float]:
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

embedding_service = FastEmbedService()
