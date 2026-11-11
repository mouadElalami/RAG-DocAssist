import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "embedding_provider" in data

def test_config_endpoint():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "llm_provider" in data
    assert "chunk_size" in data

def test_index_raw_text_and_search():
    payload = {
        "title": "Document de Test API",
        "text": "Le framework FastAPI permet de concevoir des APIs REST performantes avec validation automatique Pydantic."
    }
    res_index = client.post("/api/documents/index-text", json=payload)
    assert res_index.status_code == 200
    data_index = res_index.json()
    assert data_index["success"] is True
    assert data_index["chunks_created"] >= 1

    # Search for it
    search_payload = {
        "query": "FastAPI et validation Pydantic",
        "top_k": 2
    }
    res_search = client.post("/api/search", json=search_payload)
    assert res_search.status_code == 200
    data_search = res_search.json()
    assert data_search["total_results"] >= 1
    assert "FastAPI" in data_search["results"][0]["content"]

def test_chat_endpoint():
    chat_payload = {
        "query": "De quoi parle le document FastAPI ?",
        "top_k": 2,
        "provider": "mock"
    }
    response = client.post("/api/chat", json=chat_payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
