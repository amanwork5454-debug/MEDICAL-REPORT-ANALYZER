import re
import numpy as np


class RAGPipeline:
    def __init__(self, document_processor, embeddings_manager, llm_handler, chunk_size=200, top_k=5):
        self.document_processor = document_processor
        self.embeddings_manager = embeddings_manager
        self.llm_handler = llm_handler
        self.chunk_size = chunk_size
        self.top_k = top_k
        self._chunks = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text):
        """Split text into overlapping sentence-based chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            current_chunk.append(sentence)
            current_size += len(sentence.split())

            if current_size >= self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def _extract_text(self):
        """Extract raw text from the document using DocumentProcessor."""
        path = self.document_processor.file_path
        if path.endswith('.pdf'):
            return self.document_processor.process_pdf()
        elif path.endswith('.docx'):
            return self.document_processor.process_docx()
        else:
            return self.document_processor.process_txt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(self):
        """Extract, chunk, embed, and index the document. Call once before querying."""
        text = self._extract_text()
        self._chunks = self._chunk_text(text)
        self.embeddings_manager.encode_documents(self._chunks)
        self.embeddings_manager.build_faiss_index()

    def process_query(self, query):
        """Run the full RAG pipeline for a user query and return the generated answer."""
        if not self._chunks:
            raise RuntimeError("Document has not been indexed yet. Call index_document() first.")

        # Retrieve top-k relevant chunk indices via FAISS
        try:
            indices, _ = self.embeddings_manager.search_similar_documents(query, k=self.top_k)
        except Exception as exc:
            raise RuntimeError(f"Failed to retrieve relevant document chunks: {exc}") from exc
        relevant_chunks = [self._chunks[i] for i in indices if i < len(self._chunks)]

        if not relevant_chunks:
            return "No relevant information found in the document."

        # Build context and ask the LLM
        context = "\n".join(relevant_chunks)
        return self.llm_handler.answer_medical_question(query, context=context)
