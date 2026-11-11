import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.services.embeddings import EmbeddingFactory

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = index_dir or settings.FAISS_INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.index_dir / "registry.json"
        self.chunks_file = self.index_dir / "chunks_store.json"
        
        self.embeddings = EmbeddingFactory.get_embeddings()
        self.vector_store: Optional[FAISS] = None
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.all_chunks: List[Dict[str, Any]] = []
        
        self._load_registry()
        self._load_or_initialize_index()

    def _load_registry(self):
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception as e:
                logger.error(f"Erreur chargement registre: {e}")
                self.registry = {}
        else:
            self.registry = {}

        if self.chunks_file.exists():
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    self.all_chunks = json.load(f)
            except Exception as e:
                logger.error(f"Erreur chargement chunks store: {e}")
                self.all_chunks = []
        else:
            self.all_chunks = []

    def _save_registry(self):
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.all_chunks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur sauvegarde registre: {e}")

    def _load_or_initialize_index(self):
        index_file = self.index_dir / "index.faiss"
        if index_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Index FAISS existant chargé avec succès.")
                return
            except Exception as e:
                logger.warning(f"Impossible de charger l'index FAISS existant ({e}). Recréation en cours.")

        # Recreate from stored chunks if available, else empty dummy doc
        if self.all_chunks:
            self._rebuild_index_from_stored_chunks()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        # Create minimal placeholder index that won't pollute user results
        dummy_doc = Document(
            page_content="RAG-DocAssist initialisation placeholder.",
            metadata={"doc_id": "__init__", "source": "system", "chunk_index": -1}
        )
        self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
        self.save_index()

    def _rebuild_index_from_stored_chunks(self):
        docs = [
            Document(page_content=c["content"], metadata=c["metadata"])
            for c in self.all_chunks
            if c.get("metadata", {}).get("doc_id") != "__init__"
        ]
        if docs:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self._create_empty_index()
        self.save_index()

    def save_index(self):
        if self.vector_store:
            try:
                self.vector_store.save_local(str(self.index_dir))
                self._save_registry()
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde de l'index FAISS: {e}")

    def add_documents(self, documents: List[Document], doc_metadata: Dict[str, Any]) -> int:
        """
        Adds chunked documents to the FAISS index and updates the registry.
        """
        if not documents:
            return 0

        doc_id = doc_metadata["doc_id"]
        
        # Store in registry
        self.registry[doc_id] = doc_metadata
        
        # Store chunks in memory & file backup
        for doc in documents:
            self.all_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Add to vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # If the index only contains the dummy doc, recreate cleanly
            if len(self.registry) == 1 and "__init__" not in self.registry:
                docs = [
                    Document(page_content=c["content"], metadata=c["metadata"])
                    for c in self.all_chunks
                ]
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vector_store.add_documents(documents)

        self.save_index()
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_doc_id: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Performs similarity search in FAISS.
        Returns list of (Document, score) where score is distance.
        """
        if not self.vector_store or not self.all_chunks:
            return []

        # Request more if filtering by doc_id
        fetch_k = k * 3 if filter_doc_id else k
        raw_results = self.vector_store.similarity_search_with_score(query, k=fetch_k)

        filtered_results: List[Tuple[Document, float]] = []
        for doc, score in raw_results:
            # Ignore placeholder
            if doc.metadata.get("doc_id") == "__init__":
                continue
            if filter_doc_id and doc.metadata.get("doc_id") != filter_doc_id:
                continue
            
            filtered_results.append((doc, float(score)))
            if len(filtered_results) >= k:
                break

        return filtered_results

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes a document by doc_id and rebuilds the FAISS vector index.
        """
        if doc_id not in self.registry:
            return False

        # Remove from registry
        del self.registry[doc_id]

        # Remove from chunks store
        self.all_chunks = [
            c for c in self.all_chunks
            if c.get("metadata", {}).get("doc_id") != doc_id
        ]

        # Rebuild index
        self._rebuild_index_from_stored_chunks()
        return True

    def reset(self):
        """
        Clears all indexed documents and resets FAISS index.
        """
        self.registry = {}
        self.all_chunks = []
        self._create_empty_index()
        self._save_registry()

    def get_stats(self) -> Dict[str, Any]:
        info = EmbeddingFactory.get_info()
        index_size_kb = 0.0
        index_file = self.index_dir / "index.faiss"
        if index_file.exists():
            index_size_kb = index_file.stat().st_size / 1024.0

        return {
            "total_documents": len(self.registry),
            "total_chunks": len([c for c in self.all_chunks if c.get("metadata", {}).get("doc_id") != "__init__"]),
            "index_type": "FAISS (IndexFlatL2 / IndexFlatIP)",
            "embedding_model": info.get("model_name", "all-MiniLM-L6-v2"),
            "embedding_dimension": 384,
            "storage_size_kb": round(index_size_kb, 2),
            "documents": list(self.registry.values())
        }
