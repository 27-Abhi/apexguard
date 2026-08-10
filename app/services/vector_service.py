import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from app.core.config import settings
from app.core.logging import get_logger
from app.interfaces.vector_store import BaseVectorStore

logger = get_logger(__name__)

class QdrantVectorService(BaseVectorStore):
    def __init__(self):
        self._is_persistent = False
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}...")
        try:
            self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0)
            self.client.get_collections()
            self._is_persistent = True
            logger.info(f"Qdrant connected successfully (persistent mode)")
        except Exception as e:
            logger.warning(
                f"Qdrant unreachable at {settings.QDRANT_HOST}:{settings.QDRANT_PORT} — {type(e).__name__}: {e}"
            )
            logger.warning(
                "Falling back to IN-MEMORY vector storage. Data will NOT persist across restarts! "
                "Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant"
            )
            self.client = QdrantClient(":memory:")

    @property
    def is_persistent(self) -> bool:
        return self._is_persistent

    def ensure_collection(
        self, 
        collection_name: str = settings.QDRANT_COLLECTION_NAME, 
        vector_size: int = settings.EMBEDDING_DIMENSION
    ) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            logger.info(f"Creating new collection '{collection_name}' (dim={vector_size}, metric=COSINE)")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest_models.VectorParams(
                    size=vector_size,
                    distance=rest_models.Distance.COSINE
                )
            )
        else:
            logger.debug(f"Collection '{collection_name}' already exists")

    def insert_documents(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        collection_name: str = settings.QDRANT_COLLECTION_NAME
    ) -> List[str]:
        logger.info(f"Inserting {len(chunks)} chunks into collection '{collection_name}'")
        self.ensure_collection(collection_name, len(embeddings[0]) if embeddings else settings.EMBEDDING_DIMENSION)
        
        points = []
        point_ids = []
        for i, (chunk, vector, meta) in enumerate(zip(chunks, embeddings, metadatas)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            payload = {
                "text": chunk,
                **meta
            }
            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
            
        if points:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Upserted {len(points)} points → collection '{collection_name}'")
        else:
            logger.warning("No points to insert — empty chunks/embeddings")
        return point_ids

    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: str = settings.QDRANT_COLLECTION_NAME
    ) -> List[Dict[str, Any]]:
        logger.info(f"Searching collection '{collection_name}' (top_k={top_k}, filter={metadata_filter})")
        self.ensure_collection(collection_name)
        
        qdrant_filter = None
        if metadata_filter:
            must_conditions = []
            for key, val in metadata_filter.items():
                must_conditions.append(
                    rest_models.FieldCondition(
                        key=key,
                        match=rest_models.MatchValue(value=val)
                    )
                )
            if must_conditions:
                qdrant_filter = rest_models.Filter(must=must_conditions)

        search_response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        )

        results = []
        for hit in search_response.points:
            results.append({
                "id": str(hit.id),
                "score": float(hit.score),
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            })

        logger.info(f"Search returned {len(results)} results (top score: {results[0]['score']:.4f})" if results else "Search returned 0 results")
        return results

    def get_all_documents(self, collection_name: str = settings.QDRANT_COLLECTION_NAME) -> List[Dict[str, Any]]:
        logger.info(f"Scrolling all documents from collection '{collection_name}'")
        try:
            self.ensure_collection(collection_name)
            scroll_res = self.client.scroll(collection_name=collection_name, limit=100, with_payload=True)
            points, _ = scroll_res
            logger.info(f"Retrieved {len(points)} document chunks")
            return [
                {
                    "id": str(p.id),
                    "text": p.payload.get("text", ""),
                    "metadata": {k: v for k, v in p.payload.items() if k != "text"}
                }
                for p in points
            ]
        except Exception as e:
            logger.error(f"Failed to scroll documents: {type(e).__name__}: {e}")
            return []

vector_service = QdrantVectorService()
