"""Unit tests for embeddings module."""

import pytest
from unittest.mock import Mock, patch
from src.embeddings import HFEmbeddings


class TestHFEmbeddings:
    """Test cases for HFEmbeddings class."""
    
    # Test 1: Initialize with default model
    def test_init_default_model(self):
        emb = HFEmbeddings()
        
        assert emb.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    # Test 2: Embed documents
    @patch("src.embeddings.HuggingFaceEmbeddings")
    def test_embed_documents(self, mock_hf):
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_hf.return_value = mock_instance
        
        emb = HFEmbeddings()
        result = emb.embed_documents(["text1", "text2"])
        
        assert len(result) == 2
    # Test 3: Embed query
    
    @patch("src.embeddings.HuggingFaceEmbeddings")
    def test_embed_query(self, mock_hf):
        mock_instance = Mock()
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_hf.return_value = mock_instance
        
        emb = HFEmbeddings()
        result = emb.embed_query("test query")
        
        assert result == [0.1, 0.2, 0.3]