from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "RAG-DocAssist"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    FAISS_INDEX_DIR: Path = BASE_DIR / "data" / "faiss_index"
    SAMPLE_DOCS_DIR: Path = BASE_DIR / "data" / "sample_documents"
    
    # Vector Chunking parameters
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 120
    
    # Embeddings configuration
    # Options: "sentence-transformers", "huggingface", "openai", "gemini", "mock"
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # LLM Provider configuration
    # Options: "gemini", "openai", "ollama", "mock"
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    
    # Retrieval configuration
    DEFAULT_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.0  # Min score threshold if applicable
    
    # Server configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
settings.SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
