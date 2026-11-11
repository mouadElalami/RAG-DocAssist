import pytest
from langchain_core.documents import Document
from app.services.text_splitter import TextSplitterService

def test_chunking_splits_correctly():
    splitter = TextSplitterService(chunk_size=100, chunk_overlap=20)
    
    long_text = "Le découpage vectoriel (chunking) est une étape fondamentale des systèmes RAG. " * 5
    doc = Document(page_content=long_text, metadata={"doc_id": "test_1", "source": "test.txt"})
    
    chunks = splitter.split_documents([doc])
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 120
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["doc_id"] == "test_1"

def test_chunking_short_document():
    splitter = TextSplitterService(chunk_size=500, chunk_overlap=50)
    short_text = "Court texte d'introduction."
    doc = Document(page_content=short_text, metadata={"doc_id": "short_1", "source": "court.txt"})
    
    chunks = splitter.split_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == short_text
    assert chunks[0].metadata["chunk_index"] == 0
