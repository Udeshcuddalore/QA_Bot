"""Unit tests for vector_store module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """Test cases for VectorStoreManager class."""
    
    # Test 1: Initialize VectorStoreManager
    def test_init(self):
        mock_embeddings = Mock()
        
        vm = VectorStoreManager(mock_embeddings)
        
        assert vm.embeddings == mock_embeddings
        assert vm.vector_store is None
    
    # Test 2: Create FAISS vector store
    @patch("src.vector_store.FAISS")
    def test_create_faiss_store(self, mock_faiss):
        mock_embeddings = Mock()
        mock_chunks = [Mock(), Mock()]
        
        # Setup mock
        mock_store = Mock()
        mock_faiss.from_documents.return_value = mock_store
        
        vm = VectorStoreManager(mock_embeddings)
        result = vm.create_store(mock_chunks, "FAISS")
        
        mock_faiss.from_documents.assert_called_once()
        assert result == mock_store
    
    # Test 3: Create Chroma vector store
    @patch("src.vector_store.Chroma")
    def test_create_chroma_store(self, mock_chroma):
        mock_embeddings = Mock()
        mock_chunks = [Mock(), Mock()]
        
        # Setup mock
        mock_store = Mock()
        mock_chroma.from_documents.return_value = mock_store
        
        vm = VectorStoreManager(mock_embeddings)
        result = vm.create_store(mock_chunks, "Chroma")
        
        mock_chroma.from_documents.assert_called_once()
        assert result == mock_store