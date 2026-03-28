from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreHandler:

    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None

    def build_index(self, documents):
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )

    def get_retriever(self, k=4):
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def similarity_search(self, query, k=4):
        return self.vectorstore.similarity_search(query, k=k)
    
    def clear(self):
        del self.vectorstore
        self.vectorstore = None