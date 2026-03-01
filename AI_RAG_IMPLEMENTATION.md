# Endee RAG Integration Project

\# Endee-Powered Semantic Search RAG System



\## Overview



This fork extends the Endee repository with a FastAPI-based Retrieval-Augmented Generation (RAG) style semantic search system.



The system allows:



\- Uploading text documents

\- Chunking documents into smaller segments

\- Generating embeddings using Sentence Transformers

\- Storing embeddings via a pluggable Vector Store abstraction

\- Performing semantic similarity search



This implementation is designed to integrate with Endee as the production vector database.



---



\## Architecture



User → FastAPI → RAGPipeline → VectorStore → EmbeddingModel



\### Components



\### 1. EmbeddingModel

\- Uses `sentence-transformers/all-MiniLM-L6-v2`

\- Converts text into dense 384-dimensional vectors



\### 2. Vector Store Abstraction



A modular vector storage layer was introduced:



\- `InMemoryVectorStore` (development mode)

\- `EndeeVectorStore` (production-ready integration layer)



This abstraction allows seamless switching between local memory and Endee.



Example:



```python

self.\_store = InMemoryVectorStore()

\# Swap with EndeeVectorStore() for production

