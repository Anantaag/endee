from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text: str):
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]):
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]