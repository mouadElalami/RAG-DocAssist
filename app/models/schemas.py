from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    source: str
    filename: str
    file_type: str
    file_size_bytes: int
    total_chunks: int
    indexed_at: datetime
    doc_id: str

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: Optional[int] = None
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RawTextIndexRequest(BaseModel):
    title: str = Field(..., description="Document or note title")
    text: str = Field(..., description="Raw text content to split and index")
    chunk_size: Optional[int] = Field(None, description="Optional custom chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Optional custom chunk overlap")

class IndexResponse(BaseModel):
    success: bool
    message: str
    doc_id: str
    filename: str
    chunks_created: int
    total_indexed_chunks: int
    time_taken_ms: float

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")
    score_threshold: Optional[float] = Field(default=None, description="Maximum distance or threshold")
    filter_doc_id: Optional[str] = Field(default=None, description="Optional filter by document ID")

class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    content: str
    score: float = Field(..., description="Distance or similarity score (lower distance = closer)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    query: str
    total_results: int
    time_taken_ms: float
    results: List[SearchResultItem]

class Citation(BaseModel):
    citation_id: int
    source: str
    doc_id: str
    chunk_index: int
    page: Optional[int] = None
    relevance_score: float
    excerpt: str

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve")
    provider: Optional[str] = Field(default=None, description="Override LLM provider ('gemini', 'openai', 'ollama', 'mock')")
    api_key: Optional[str] = Field(default=None, description="Optional temporary API key passed from UI")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Creativity temperature")
    system_prompt: Optional[str] = Field(default=None, description="Custom system instruction for RAG")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Previous conversation messages")

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[Citation]
    provider: str
    model: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

class VectorStoreStats(BaseModel):
    total_documents: int
    total_chunks: int
    index_type: str
    embedding_model: str
    embedding_dimension: int
    storage_size_kb: float
    documents: List[DocumentMetadata]

class ConfigUpdateRequest(BaseModel):
    llm_provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    default_top_k: Optional[int] = None

class SystemHealth(BaseModel):
    status: str
    app_name: str
    version: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    faiss_initialized: bool
    total_indexed_chunks: int
