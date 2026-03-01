"""RAG pipeline using pluggable vector store (Endee-ready architecture)."""

import numpy as np
from app.embeddings import EmbeddingModel
from app.utils import chunk_text, load_text_file
from app.vector_store import InMemoryVectorStore  # can switch to EndeeVectorStore later


class RAGPipeline:
    """Index documents and search using embeddings + vector store abstraction."""

    def __init__(self):
        self._model = EmbeddingModel()
        self._store = InMemoryVectorStore()  # swap with EndeeVectorStore for production

    def index_document(self, file_path: str) -> None:
        """
        Load a text file, chunk it, embed each chunk,
        and store embeddings in vector store.
        """
        text = load_text_file(file_path)
        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = np.array(self._model.embed_text(chunk))
            self._store.add(chunk, embedding)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        Embed query and retrieve top_k relevant chunks
        from vector store.
        """
        query_embedding = np.array(self._model.embed_text(query))
        return self._store.search(query_embedding, top_k)