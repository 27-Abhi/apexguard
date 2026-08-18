from typing import List, Dict, Any, Optional
from app.interfaces.embeddings import BaseEmbeddingService
from app.interfaces.vector_store import BaseVectorStore
from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

logger = get_logger(__name__)

class RAGRetrieverService:
    def __init__(
        self, 
        embedder: BaseEmbeddingService = embedding_service, 
        vector_store: BaseVectorStore = vector_service
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("RAG Retriever service initialized")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving context for query: '{query[:100]}' (top_k={top_k}, filter={metadata_filter})")
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        logger.info(f"Retrieved {len(results)} context chunks")
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = settings.HYBRID_TOP_K,
        metadata_filter: Optional[Dict[str, Any]] = None,
        fusion_weight: float = settings.HYBRID_FUSION_WEIGHT,
        enable_rerank: bool = settings.HYBRID_ENABLE_RERANK,
        rrf_candidate_k: int = settings.HYBRID_RRF_CANDIDATE_K,
        dense_candidate_k: int = settings.HYBRID_DENSE_CANDIDATE_K,
        sparse_candidate_k: int = settings.HYBRID_SPARSE_CANDIDATE_K
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"Hybrid retrieve called (top_k={top_k}, dense_candidate_k={dense_candidate_k}, "
            f"sparse_candidate_k={sparse_candidate_k}, rrf_candidate_k={rrf_candidate_k}, "
            f"enable_rerank={enable_rerank})"
        )
        dense_results = self.retrieve(query, dense_candidate_k, metadata_filter)
        sparse_results = self.vector_store.sparse_search(query, sparse_candidate_k, metadata_filter)
        fused = self._rrf_fuse(dense_results, sparse_results, fusion_weight)[:rrf_candidate_k]

        if enable_rerank:
            reranked = self._rerank(query, fused)
            if reranked:
                fused = reranked

        return fused[:top_k]

    def _rrf_fuse(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        fusion_weight: float
    ) -> List[Dict[str, Any]]:
        dense_weight = max(0.0, min(1.0, fusion_weight))
        sparse_weight = 1.0 - dense_weight
        by_id: Dict[str, Dict[str, Any]] = {}

        for rank, result in enumerate(dense_results, start=1):
            item = by_id.setdefault(result["id"], {**result, "score": 0.0})
            item["score"] += dense_weight / (settings.HYBRID_RRF_K + rank)
            item.setdefault("metadata", {})["dense_score"] = result["score"]

        for rank, result in enumerate(sparse_results, start=1):
            item = by_id.setdefault(result["id"], {**result, "score": 0.0})
            item["score"] += sparse_weight / (settings.HYBRID_RRF_K + rank)
            item.setdefault("metadata", {})["sparse_score"] = result["score"]

        return sorted(by_id.values(), key=lambda item: item["score"], reverse=True)

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            ranker = Ranker(model_name=settings.RERANK_MODEL)
            passages = [
                {"id": result["id"], "text": result["text"], "meta": result["metadata"]}
                for result in results
            ]
            ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
            by_id = {result["id"]: result for result in results}
            reranked = []
            for item in ranked:
                result = by_id.get(item["id"])
                if result:
                    result = {**result, "score": float(item.get("score", result["score"]))}
                    reranked.append(result)
            return reranked
        except Exception as e:
            logger.warning(f"Rerank unavailable; returning RRF ranking: {type(e).__name__}: {e}")
            return results

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, res in enumerate(search_results, start=1):
            source_doc = res.get("metadata", {}).get("filename", "Unknown document")
            formatted.append(f"[Source {i}: {source_doc}]\n{res['text']}")
        context = "\n\n".join(formatted)
        logger.debug(f"Formatted context from {len(search_results)} sources ({len(context)} chars)")
        return context

retriever_service = RAGRetrieverService()
