# Endee-Powered Semantic Search RAG System

## Overview

This fork extends the Endee repository with a FastAPI-based Retrieval-Augmented Generation (RAG) style semantic search system.

The system allows:

- Uploading text documents
- Chunking documents into smaller segments
- Generating embeddings using Sentence Transformers
- Storing embeddings via a pluggable Vector Store abstraction
- Performing semantic similarity search

This implementation is designed to integrate with Endee as the production vector database.

---

## Architecture

User → FastAPI → RAGPipeline → VectorStore → EmbeddingModel

### Components

### 1. EmbeddingModel
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Converts text into dense 384-dimensional vectors

### 2. Vector Store Abstraction

A modular vector storage layer was introduced:

- `InMemoryVectorStore` (development mode)
- `EndeeVectorStore` (production-ready integration layer)

This abstraction allows seamless switching between local memory and Endee.

Example:

```python
self._store = InMemoryVectorStore()
# Swap with EndeeVectorStore() for production

### 1. How Endee Would Be Used:

In production mode:

1. Endee runs via Docker.

2. EndeeVectorStore connects to Endee through its REST API.

3. During indexing:

     Each chunk embedding is inserted into an Endee collection.

4. During search:

    Query embeddings are sent to Endee.

    Endee performs nearest neighbor search.

    Matching chunks are returned to the user.

This design ensures clean separation between:

  Embedding generation

  Storage layer

  Retrieval logic

### Running the Project: 

# Install Dependencies
pip install -r requirements.txt

# Run FastAPI Server
python -m uvicorn app.main:app --reload

### Open:
http://127.0.0.1:8000/docs



### Example Use Cases:

1. Semantic academic notes search

2. Knowledge retrieval systems

3. RAG-based LLM augmentation

4. AI-powered document querying

### Future Improvements

   Full Endee REST API integration
   Persistent vector collections
   Metadata filtering
   LLM answer generation layer on top of retrieval

# 🚀 Then Commit Properly

Run:

# ```bash
git add AI_RAG_IMPLEMENTATION.md
git commit -m "Clean RAG implementation documentation"
git push