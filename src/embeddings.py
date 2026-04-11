from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingsManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None

    def encode_documents(self, documents):
        self.embeddings = self.model.encode(documents, convert_to_numpy=True).astype('float32')

    def build_faiss_index(self):
        if self.embeddings is not None and self.embeddings.shape[0] > 0:
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def search_similar_documents(self, query, k=5):
        if self.index is None:
            raise RuntimeError("FAISS index has not been built. Call build_faiss_index() first.")
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        D, I = self.index.search(query_embedding, k)
        return I[0], D[0]  # return indices and distances
