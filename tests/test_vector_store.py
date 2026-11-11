import shutil
from pathlib import Path
import pytest
from langchain_core.documents import Document
from app.services.vector_store import VectorStoreService

@pytest.fixture
def temp_vector_store(tmp_path):
    store = VectorStoreService(index_dir=tmp_path / "faiss_test")
    yield store
    # Cleanup
    if (tmp_path / "faiss_test").exists():
        shutil.rmtree(tmp_path / "faiss_test")

def test_add_and_search_documents(temp_vector_store):
    docs = [
        Document(
            page_content="L'intelligence artificielle générative et les LLMs révolutionnent le travail documentaire.",
            metadata={"doc_id": "doc1", "source": "ia_info.txt", "chunk_index": 0}
        ),
        Document(
            page_content="La recette du gâteau au chocolat nécessite des œufs, de la farine et du beurre.",
            metadata={"doc_id": "doc2", "source": "recette.txt", "chunk_index": 0}
        )
    ]
    meta = {
        "doc_id": "doc1",
        "source": "ia_info.txt",
        "filename": "ia_info.txt",
        "file_type": "txt",
        "file_size_bytes": 100,
        "total_chunks": 1,
        "indexed_at": "2024-01-01T00:00:00"
    }
    temp_vector_store.add_documents([docs[0]], meta)
    
    meta2 = {
        "doc_id": "doc2",
        "source": "recette.txt",
        "filename": "recette.txt",
        "file_type": "txt",
        "file_size_bytes": 80,
        "total_chunks": 1,
        "indexed_at": "2024-01-01T00:00:00"
    }
    temp_vector_store.add_documents([docs[1]], meta2)
    
    # Query related to IA
    results = temp_vector_store.similarity_search_with_score("intelligence artificielle et LLM", k=1)
    assert len(results) > 0
    top_doc, score = results[0]
    assert top_doc.metadata["doc_id"] == "doc1"
    assert "générative" in top_doc.page_content

def test_delete_document(temp_vector_store):
    doc = Document(
        page_content="Document à supprimer.",
        metadata={"doc_id": "del_me", "source": "temp.txt", "chunk_index": 0}
    )
    meta = {
        "doc_id": "del_me",
        "source": "temp.txt",
        "filename": "temp.txt",
        "file_type": "txt",
        "file_size_bytes": 20,
        "total_chunks": 1,
        "indexed_at": "2024-01-01T00:00:00"
    }
    temp_vector_store.add_documents([doc], meta)
    assert "del_me" in temp_vector_store.registry

    temp_vector_store.delete_document("del_me")
    assert "del_me" not in temp_vector_store.registry
    results = temp_vector_store.similarity_search_with_score("supprimer", k=2)
    assert not any(d.metadata.get("doc_id") == "del_me" for d, _ in results)
