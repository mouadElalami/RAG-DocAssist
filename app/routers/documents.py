import time
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.config import settings
from app.models.schemas import (
    IndexResponse,
    RawTextIndexRequest,
    VectorStoreStats,
    DocumentMetadata
)
from app.services.document_loader import DocumentLoaderService
from app.services.text_splitter import TextSplitterService
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/api/documents", tags=["Documents & Vectorisation"])

# Vector Store Dependency singleton
_vector_store = VectorStoreService()

def get_vector_store() -> VectorStoreService:
    return _vector_store


@router.post("/upload", response_model=List[IndexResponse], summary="Téléversement et vectorisation de documents")
async def upload_documents(
    files: List[UploadFile] = File(..., description="Fichiers PDF, DOCX, TXT, MD"),
    chunk_size: int = Form(default=settings.CHUNK_SIZE, description="Taille des segments de texte"),
    chunk_overlap: int = Form(default=settings.CHUNK_OVERLAP, description="Chevauchement des segments"),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Téléverse un ou plusieurs documents, extrait le texte, applique le découpage vectoriel (chunking)
    et indexe les embeddings dans la base FAISS.
    """
    responses = []
    splitter = TextSplitterService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for file in files:
        start_time = time.perf_counter()
        original_filename = file.filename or "unknown_file"
        file_path = settings.UPLOAD_DIR / original_filename

        try:
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Load document(s)
            docs, doc_id = DocumentLoaderService.load_file(file_path, original_filename)

            # Split into chunks
            chunks = splitter.split_documents(docs)

            # Save in vector store
            metadata = {
                "source": original_filename,
                "filename": original_filename,
                "file_type": Path(original_filename).suffix.lstrip(".").lower(),
                "file_size_bytes": file_path.stat().st_size,
                "total_chunks": len(chunks),
                "indexed_at": docs[0].metadata.get("indexed_at"),
                "doc_id": doc_id
            }
            vector_store.add_documents(chunks, metadata)

            time_taken = round((time.perf_counter() - start_time) * 1000, 2)
            stats = vector_store.get_stats()

            responses.append(
                IndexResponse(
                    success=True,
                    message=f"Le document '{original_filename}' a été découpé en {len(chunks)} segments et indexé.",
                    doc_id=doc_id,
                    filename=original_filename,
                    chunks_created=len(chunks),
                    total_indexed_chunks=stats["total_chunks"],
                    time_taken_ms=time_taken
                )
            )
        except Exception as e:
            time_taken = round((time.perf_counter() - start_time) * 1000, 2)
            responses.append(
                IndexResponse(
                    success=False,
                    message=f"Erreur lors du traitement de '{original_filename}': {str(e)}",
                    doc_id="",
                    filename=original_filename,
                    chunks_created=0,
                    total_indexed_chunks=vector_store.get_stats()["total_chunks"],
                    time_taken_ms=time_taken
                )
            )
        finally:
            file.file.close()

    return responses


@router.post("/index-text", response_model=IndexResponse, summary="Indexation directe de texte brut")
async def index_raw_text(
    payload: RawTextIndexRequest,
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Permet d'indexer directement un texte brut (note, extrait, article) sans téléverser de fichier physique.
    """
    start_time = time.perf_counter()
    try:
        docs, doc_id = DocumentLoaderService.load_raw_text(payload.title, payload.text)
        
        splitter = TextSplitterService(
            chunk_size=payload.chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=payload.chunk_overlap or settings.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)

        metadata = {
            "source": payload.title,
            "filename": f"{payload.title}.txt",
            "file_type": "text_note",
            "file_size_bytes": len(payload.text.encode("utf-8")),
            "total_chunks": len(chunks),
            "indexed_at": docs[0].metadata.get("indexed_at"),
            "doc_id": doc_id
        }
        vector_store.add_documents(chunks, metadata)
        time_taken = round((time.perf_counter() - start_time) * 1000, 2)
        stats = vector_store.get_stats()

        return IndexResponse(
            success=True,
            message=f"La note '{payload.title}' a été indexée avec succès ({len(chunks)} chunks).",
            doc_id=doc_id,
            filename=payload.title,
            chunks_created=len(chunks),
            total_indexed_chunks=stats["total_chunks"],
            time_taken_ms=time_taken
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'indexation: {str(e)}")


@router.get("", response_model=VectorStoreStats, summary="Statistiques de la base vectorielle FAISS")
async def list_documents(vector_store: VectorStoreService = Depends(get_vector_store)):
    """
    Retourne la liste des documents indexés ainsi que les métriques de l'index FAISS.
    """
    stats = vector_store.get_stats()
    return VectorStoreStats(**stats)


@router.delete("/{doc_id}", summary="Supprimer un document de l'index")
async def delete_document(
    doc_id: str,
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Supprime un document par son identifiant unique et reconstruit l'index vectoriel.
    """
    deleted = vector_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' introuvable.")
    return {"success": True, "message": f"Document '{doc_id}' supprimé et index réorganisé."}


@router.delete("", summary="Réinitialiser l'ensemble de la base vectorielle")
async def reset_all_documents(vector_store: VectorStoreService = Depends(get_vector_store)):
    """
    Efface tous les documents et réinitialise l'index FAISS à zéro.
    """
    vector_store.reset()
    return {"success": True, "message": "Base vectorielle FAISS réinitialisée avec succès."}


@router.post("/load-samples", summary="Charger des documents d'exemple prédéfinis")
async def load_sample_documents(vector_store: VectorStoreService = Depends(get_vector_store)):
    """
    Charge automatiquement les documents d'exemple fournis pour une démonstration instantanée.
    """
    splitter = TextSplitterService()
    results = []

    for file_path in settings.SAMPLE_DOCS_DIR.glob("*.*"):
        if file_path.suffix.lower() in DocumentLoaderService.SUPPORTED_EXTENSIONS:
            try:
                docs, doc_id = DocumentLoaderService.load_file(file_path, file_path.name)
                chunks = splitter.split_documents(docs)
                metadata = {
                    "source": file_path.name,
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lstrip(".").lower(),
                    "file_size_bytes": file_path.stat().st_size,
                    "total_chunks": len(chunks),
                    "indexed_at": docs[0].metadata.get("indexed_at"),
                    "doc_id": doc_id
                }
                vector_store.add_documents(chunks, metadata)
                results.append(file_path.name)
            except Exception as e:
                pass

    return {
        "success": True,
        "message": f"{len(results)} document(s) d'exemple chargé(s) avec succès.",
        "documents": results
    }
