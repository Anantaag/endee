from typing import List, Dict
import numpy as np


class BaseVectorStore:
    def add(self, text: str, embedding: np.ndarray):
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self):
        self._store: List[Dict] = []

    def add(self, text: str, embedding: np.ndarray):
        self._store.append({"text": text, "embedding": embedding})

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        if not self._store:
            return []

        similarities = []
        for item in self._store:
            emb = item["embedding"]
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-9
            )
            similarities.append(sim)

        indices = np.argsort(similarities)[::-1][:top_k]
        return [self._store[i]["text"] for i in indices]


class EndeeVectorStore(BaseVectorStore):
    """
    Placeholder for real Endee integration.
    In production, this would:
    - Connect to Endee REST API
    - Insert vectors into a collection
    - Query nearest neighbors
    """

    def __init__(self):
        # Example:
        # self.base_url = "http://localhost:8080"
        pass

    def add(self, text: str, embedding: np.ndarray):
        # Here we would POST to Endee API
        # requests.post(...)
        pass

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        # Here we would call Endee nearest neighbor endpoint
        # response = requests.post(...)
        # return parsed texts
        return []