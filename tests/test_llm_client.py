"""Unit tests for llm_client module."""

import pytest
from unittest.mock import Mock, patch
from src.llm_client import LLMClient


class TestLLMClient:
    """Test cases for LLMClient class."""
    
    # Test 1: Initialize with Ollama
    @patch("src.llm_client.ChatOllama")
    def test_init_ollama(self, mock_chat):
        mock_instance = Mock()
        mock_chat.return_value = mock_instance
        
        client = LLMClient(provider="ollama", model_name="llama3")
        
        assert client.provider == "ollama"
        assert client.model_name == "llama3"
        mock_chat.assert_called_once()
    
    # Test 2: Initialize with OpenAI
    @patch("src.llm_client.ChatOpenAI")
    def test_init_openai(self, mock_chat):
        mock_instance = Mock()
        mock_chat.return_value = mock_instance
        
        client = LLMClient(provider="openai", model_name="gpt-4")
        
        assert client.provider == "openai"
        assert client.model_name == "gpt-4"
        mock_chat.assert_called_once()
    
    # Test 3: Get LLM object
    @patch("src.llm_client.ChatOllama")
    def test_get_llm(self, mock_chat):
        mock_instance = Mock()
        mock_chat.return_value = mock_instance
        
        client = LLMClient(provider="ollama", model_name="llama3")
        result = client.get_llm()
        
        assert result == mock_instance