from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loguru import logger
from typing import List
 
 
class VectorStoreManager:
    """Handles vector database operations"""
    
    def __init__(self, embeddings):
        """
        Initialize vector store
        
        Args:
            embeddings: Embedding object from EmbeddingManager
        """
        self.embeddings = embeddings
        self.vector_store = None
    
    def create_store(self, chunks: List[Document],vectordbtype:str):
        """
        Create vector store from document chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            FAISS vector store
        """
        if not chunks:
            logger.warning("No chunks to create vector store")
            return None
        
        logger.info(f"Creating vector store with {len(chunks)} chunks")
        if vectordbtype == "FAISS":
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
        else:
            self.vector_store = Chroma.from_documents(
        documents=chunks,embedding=self.embeddings
        )

        logger.success("Vector store created successfully")
        return self.vector_store
    
    def get_retriever(self, k: int = 5):
        """
        Get retriever from vector store
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized. Call create_store first.")
        
        logger.info(f"Creating retriever (k={k})")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
 