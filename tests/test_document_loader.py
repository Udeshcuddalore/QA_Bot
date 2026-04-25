
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.document_loader import DocumentLoader


class TestDocumentLoader:
    """Simple test cases for DocumentLoader."""
    
    # Test 1: Initialize with single file
    def test_init_single_file(self):
        file = Mock()
        file.name = "test.pdf"
        
        loader = DocumentLoader(file)
        
        assert len(loader.files) == 1
    
    # Test 2: Initialize with list of files
    def test_init_list_of_files(self):
        files = [Mock(name="file1.pdf"), Mock(name="file2.pdf")]
        
        loader = DocumentLoader(files)
        
        assert len(loader.files) == 2
    
    # Test 3: Initialize with empty list
    def test_init_empty_list(self):
        loader = DocumentLoader([])
        
        assert len(loader.files) == 0
    
    # Test 4: Non-PDF file is rejected
    def test_reject_non_pdf(self):
        file = Mock()
        file.name = "test.txt"
        
        loader = DocumentLoader(file)
        result = loader.PDF_loader()
        
        assert result == []
    
    # Test 5: File without name attribute is rejected
    def test_reject_file_without_name(self):
        file = Mock(spec=[])  # No attributes
        
        loader = DocumentLoader(file)
        result = loader.PDF_loader()
        
        assert result == []
    
    
    # Test 7: Multiple files loading
    def test_multiple_files(self):
        files = [Mock(name="file1.pdf"), Mock(name="file2.pdf")]
        
        loader = DocumentLoader(files)
        
        with patch("src.document_loader.PyPDFLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load.return_value = [Mock()]
            mock_loader_class.return_value = mock_loader
            
            result = loader.PDF_loader()
            
            assert mock_loader_class.call_count == 2
            assert len(result) == 2
    
    # Test 8: Exception handling
    def test_exception_handling(self):
        file = Mock()
        file.name = "test.pdf"
        
        loader = DocumentLoader(file)
        
        with patch("src.document_loader.PyPDFLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load.side_effect = Exception("Error")
            mock_loader_class.return_value = mock_loader
            
            result = loader.PDF_loader()
            
            assert result == []
    
    # Test 9: Uppercase PDF extension
    def test_uppercase_pdf(self):
        file = Mock()
        file.name = "test.PDF"
        
        loader = DocumentLoader(file)
        
        with patch("src.document_loader.PyPDFLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load.return_value = []
            mock_loader_class.return_value = mock_loader
            
            result = loader.PDF_loader()
            
            assert isinstance(result, list)
    
    # Test 10: File not found
    def test_file_not_found(self):
        file = Mock()
        file.name = "/nonexistent/file.pdf"
        
        loader = DocumentLoader(file)
        result = loader.PDF_loader()
        
        assert result == []