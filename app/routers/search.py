import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.services.vector_store import VectorStoreService
from app.routers.documents import get_vector_store

router = APIRouter(prefix="/api/search", tags=["Recherche Sémantique"])


@router.post("", response_model=SearchResponse, summary="Recherche sémantique vectorielle directe")
async def semantic_search(
    payload: SearchRequest,
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Exécute une recherche par similarité dans l'index FAISS à partir du texte de requête.
    Retourne les segments de texte les plus pertinents avec leurs scores de distance et métadonnées.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="La requête de recherche ne peut pas être vide.")

    start_time = time.perf_counter()
    raw_results = vector_store.similarity_search_with_score(
        query=payload.query,
        k=payload.top_k,
        filter_doc_id=payload.filter_doc_id
    )
    time_taken = round((time.perf_counter() - start_time) * 1000, 2)

    items: List[SearchResultItem] = []
    for doc, score in raw_results:
        if payload.score_threshold is not None and score > payload.score_threshold:
            continue

        items.append(
            SearchResultItem(
                chunk_id=doc.metadata.get("chunk_id", "c0"),
                doc_id=doc.metadata.get("doc_id", "doc"),
                source=doc.metadata.get("source", "Inconnu"),
                content=doc.page_content,
                score=round(score, 4),
                metadata=doc.metadata
            )
        )

    return SearchResponse(
        query=payload.query,
        total_results=len(items),
        time_taken_ms=time_taken,
        results=items
    )
