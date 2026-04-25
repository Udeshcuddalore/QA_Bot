import pytest 
from src.text_splitter import split_text
from langchain_core.documents import Document

class TestTextSplitter:
    """ Simple test cases from split_text_function."""
    
    #Test 1: Empty data returns empty lists
    def test_empty_data(self):
        result = split_text([])

        assert result == []
    
    # Test 2: Split documents returns chunks
    def test_split_documents(self):
        #create mock documents
        docs = [
            Document(
                page_content="This is a long text that needs to be split into smaller chunks for processing.",
                metadata={"source": "test.pdf"}
            ),
            Document(
                page_content="Another document with some content here.",
                metadata={"source": "test2.pdf"}
            )
        ]
        result = split_text(docs, chunk_size=20, chunk_overlap=5)
        
        # Should return list of chunks
        assert isinstance(result, list)
        assert len(result) > 0

    def test_custom_chunk_size(self):
        docs = [
            Document(
                page_content="Short text",
                metadata={"source": "test.pdf"}
            )
        ]
        
        result = split_text(docs, chunk_size=100, chunk_overlap=10)
        
        # Should return list
        assert isinstance(result, list)