from fastapi import APIRouter, Depends
from app.config import settings
from app.models.schemas import SystemHealth, ConfigUpdateRequest
from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingFactory
from app.routers.documents import get_vector_store

router = APIRouter(prefix="/api", tags=["Système & Configuration"])


@router.get("/health", response_model=SystemHealth, summary="Vérification de l'état de l'application")
async def health_check(vector_store: VectorStoreService = Depends(get_vector_store)):
    info = EmbeddingFactory.get_info()
    stats = vector_store.get_stats()
    
    return SystemHealth(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        embedding_provider=info.get("provider", settings.EMBEDDING_PROVIDER),
        embedding_model=info.get("model_name", settings.EMBEDDING_MODEL_NAME),
        llm_provider=settings.LLM_PROVIDER,
        faiss_initialized=vector_store.vector_store is not None,
        total_indexed_chunks=stats["total_chunks"]
    )


@router.get("/config", summary="Consulter la configuration active")
async def get_configuration():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "gemini_model": settings.GEMINI_MODEL,
        "openai_model": settings.OPENAI_MODEL,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.OLLAMA_MODEL,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "default_top_k": settings.DEFAULT_TOP_K
    }


@router.post("/config", summary="Mettre à jour la configuration à chaud")
async def update_configuration(payload: ConfigUpdateRequest):
    if payload.llm_provider:
        settings.LLM_PROVIDER = payload.llm_provider
    if payload.gemini_api_key is not None:
        settings.GEMINI_API_KEY = payload.gemini_api_key or None
    if payload.openai_api_key is not None:
        settings.OPENAI_API_KEY = payload.openai_api_key or None
    if payload.chunk_size:
        settings.CHUNK_SIZE = payload.chunk_size
    if payload.chunk_overlap:
        settings.CHUNK_OVERLAP = payload.chunk_overlap
    if payload.default_top_k:
        settings.DEFAULT_TOP_K = payload.default_top_k

    return {
        "success": True,
        "message": "Configuration mise à jour avec succès.",
        "active_provider": settings.LLM_PROVIDER,
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP
    }
