# RAG Pipeline Implementation

class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def process_query(self, query):
        # Step 1: Use retriever to fetch relevant documents
        documents = self.retriever.retrieve(query)

        # Step 2: Generate response based on retrieved documents
        response = self.generator.generate(documents)

        return response

# Example usage:
if __name__ == '__main__':
    # Assuming we have already defined `retriever` and `generator`
    rag_pipeline = RAGPipeline(retriever, generator)
    query = "What are the symptoms of diabetes?"
    print(rag_pipeline.process_query(query))