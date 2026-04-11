from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingsManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None

    def encode_documents(self, documents):
        self.embeddings = self.model.encode(documents, convert_to_tensor=True)

    def build_faiss_index(self):
        if self.embeddings is not None and self.embeddings.shape[0] > 0:
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(np.array(self.embeddings.cpu()))

    def search_similar_documents(self, query, k=5):
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        D, I = self.index.search(np.array(query_embedding.cpu()), k)
        return I[0], D[0]  # return indices and distances
