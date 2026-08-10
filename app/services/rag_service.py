from typing import List, Dict, Any, Optional
from app.interfaces.embeddings import BaseEmbeddingService
from app.interfaces.vector_store import BaseVectorStore
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

class RAGRetrieverService:
    def __init__(
        self, 
        embedder: BaseEmbeddingService = embedding_service, 
        vector_store: BaseVectorStore = vector_service
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        return results

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, res in enumerate(search_results, start=1):
            source_doc = res.get("metadata", {}).get("filename", "Unknown document")
            formatted.append(f"[Source {i}: {source_doc}]\n{res['text']}")
        return "\n\n".join(formatted)

retriever_service = RAGRetrieverService()
