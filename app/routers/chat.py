from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.vector_store import VectorStoreService
from app.services.rag_service import RAGService
from app.routers.documents import get_vector_store

router = APIRouter(prefix="/api/chat", tags=["Question / Réponse RAG"])


@router.post("", response_model=ChatResponse, summary="Poser une question avec génération RAG & citations")
async def rag_chat(
    payload: ChatRequest,
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Pipeline RAG complet (Retrieval-Augmented Generation) :
    1. Récupération sémantique des segments documentaires les plus proches via FAISS.
    2. Construction d'un prompt enrichi avec le contexte documentaire et l'historique de conversation.
    3. Synthèse et génération de la réponse par le modèle LLM (Gemini, OpenAI, Ollama ou local).
    4. Extraction des citations et sources avec métadonnées précises.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    rag_service = RAGService(vector_store=vector_store)
    
    response = rag_service.ask(
        query=payload.query,
        top_k=payload.top_k,
        provider=payload.provider,
        api_key=payload.api_key,
        temperature=payload.temperature,
        system_prompt=payload.system_prompt,
        history=payload.history
    )
    
    return response
