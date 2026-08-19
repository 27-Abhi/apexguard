import re
from typing import List, Dict, Any, Optional
from app.interfaces.embeddings import BaseEmbeddingService
from app.interfaces.vector_store import BaseVectorStore
from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

logger = get_logger(__name__)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "what", "how", "which", "where", "who", "whom", "why",
    "can", "you", "tell", "me", "about", "do", "does", "did", "please", "show"
}

class RAGRetrieverService:
    def __init__(
        self, 
        embedder: BaseEmbeddingService = embedding_service, 
        vector_store: BaseVectorStore = vector_service
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("RAG Retriever service initialized (Data Agnostic Mode)")

    def _rewrite_query_for_sparse(self, query: str) -> str:
        """Light query rewrite for sparse retrieval: removes conversational framing."""
        cleaned = re.sub(r'^[?\s]*', '', query)
        tokens = re.findall(r'\b[A-Za-z0-9_.:/-]+\b', cleaned)
        filtered = [t for t in tokens if t.lower() not in STOP_WORDS]
        rewritten = " ".join(filtered) if filtered else query
        return rewritten

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving context for query: '{query[:100]}' (top_k={top_k})")
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = settings.HYBRID_TOP_K,
        metadata_filter: Optional[Dict[str, Any]] = None,
        fusion_weight: float = settings.HYBRID_FUSION_WEIGHT,
        enable_rerank: bool = settings.HYBRID_ENABLE_RERANK,
        # Reduced candidates to fix the 10-second Latency bottleneck
        rrf_candidate_k: int = 15,  
        dense_candidate_k: int = 20,
        sparse_candidate_k: int = 20,
        # Dynamic threshold to boost Precision
        min_rerank_score: float = 0.05 
    ) -> List[Dict[str, Any]]:
        
        # 1. Dense retrieval
        dense_results = self.retrieve(query, dense_candidate_k, metadata_filter)

        # 2. Sparse retrieval
        sparse_query = self._rewrite_query_for_sparse(query)
        sparse_results = self.vector_store.sparse_search(sparse_query, sparse_candidate_k, metadata_filter)

        # 3. Fuse scores
        fused = self._rrf_fuse(dense_results, sparse_results, fusion_weight)[:rrf_candidate_k]

        # 4. Rerank & Filter
        if enable_rerank:
            fused = self._rerank(query, fused)
            
            # Dynamic Thresholding: Drop completely irrelevant chunks to boost Precision
            filtered_fused = [res for res in fused if res.get("score", 0) >= min_rerank_score]
            
            if filtered_fused:
                fused = filtered_fused
            else:
                logger.warning(f"All chunks fell below threshold {min_rerank_score}. Returning top 1 fallback.")
                fused = fused[:1]

        return fused[:top_k]

    def _rrf_fuse(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        fusion_weight: float
    ) -> List[Dict[str, Any]]:
        """Combines RRF rank scoring for hybrid fusion."""
        dense_weight = max(0.0, min(1.0, fusion_weight))
        sparse_weight = 1.0 - dense_weight
        by_id: Dict[str, Dict[str, Any]] = {}

        # 1. RRF Rank Fusion
        for rank, result in enumerate(dense_results, start=1):
            item = by_id.setdefault(result["id"], {**result, "score": 0.0})
            item["score"] += dense_weight / (60 + rank)
            item.setdefault("metadata", {})["dense_score"] = result["score"]

        for rank, result in enumerate(sparse_results, start=1):
            item = by_id.setdefault(result["id"], {**result, "score": 0.0})
            item["score"] += sparse_weight / (60 + rank)
            item.setdefault("metadata", {})["sparse_score"] = result["score"]

        return sorted(by_id.values(), key=lambda item: item["score"], reverse=True)

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        reranked_raw = []
        try:
            from flashrank import Ranker, RerankRequest
            ranker = Ranker(model_name=settings.RERANK_MODEL)
            passages = [
                {"id": result["id"], "text": result["text"], "meta": result["metadata"]}
                for result in results
            ]
            ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
            
            by_id = {result["id"]: result for result in results}
            for item in ranked:
                result = by_id.get(item["id"])
                if result:
                    # Return pure cross-encoder score
                    reranked_raw.append({**result, "score": float(item.get("score", 0.0))})
        except Exception as e:
            logger.warning(f"FlashRank unavailable: {e}")
            reranked_raw = [dict(r) for r in results]

        return sorted(reranked_raw, key=lambda res: res["score"], reverse=True)

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, res in enumerate(search_results, start=1):
            source_doc = res.get("metadata", {}).get("filename") or res.get("metadata", {}).get("source", "Unknown document")
            section = res.get("metadata", {}).get("section")
            section_str = f" > {section}" if section else ""
            
            # Parent context window fallback for rich LLM context
            chunk_text = res.get("metadata", {}).get("parent_text") or res["text"]
            formatted.append(f"[Source {i}: {source_doc}{section_str}]\n{chunk_text}")
        context = "\n\n".join(formatted)
        return context

retriever_service = RAGRetrieverService()