from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        pass

    @abstractmethod
    def insert_documents(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        collection_name: str
    ) -> List[str]:
        pass

    @abstractmethod
    def delete_documents(
        self,
        point_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: str = ""
    ) -> bool:
        pass

    @abstractmethod
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: str = ""
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_all_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        pass
