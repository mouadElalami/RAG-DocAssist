import logging
import hashlib
from typing import List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings
from app.config import settings

logger = logging.getLogger(__name__)


class LightweightLocalEmbeddings(Embeddings):
    """
    Fast, deterministic, dependency-light embedding fallback using feature hashing & TF weighting.
    Outputs normalized 384-dimensional dense vectors.
    Ensures vector store operations work immediately in offline or minimal test environments.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        tokens = [t.lower().strip(".,!?;:\"'()[]{}") for t in text.split() if len(t) > 1]
        if not tokens:
            return [0.0] * self.dimension
        
        vec = np.zeros(self.dimension, dtype=np.float32)
        for i, token in enumerate(tokens):
            # Deterministic hash to dimension index
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            # Positional & frequency weight
            weight = 1.0 / (1.0 + 0.05 * i)
            vec[idx] += sign * weight
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class EmbeddingFactory:
    _instance: Optional[Embeddings] = None
    _provider_used: str = "unknown"
    _model_name: str = "unknown"

    @classmethod
    def get_embeddings(cls, force_provider: Optional[str] = None) -> Embeddings:
        provider = force_provider or settings.EMBEDDING_PROVIDER
        
        if cls._instance is not None and cls._provider_used == provider:
            return cls._instance

        logger.info(f"Initialisation du fournisseur d'embeddings: {provider}")

        if provider == "sentence-transformers" or provider == "huggingface":
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                model_name = settings.EMBEDDING_MODEL_NAME or "all-MiniLM-L6-v2"
                logger.info(f"Chargement HuggingFace / sentence-transformers: {model_name}...")
                cls._instance = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                cls._provider_used = "sentence-transformers"
                cls._model_name = model_name
                return cls._instance
            except Exception as e:
                logger.warning(f"Impossible de charger HuggingFaceEmbeddings ({e}). Basculement vers LightweightLocalEmbeddings.")

        elif provider == "openai":
            if settings.OPENAI_API_KEY:
                try:
                    from langchain_community.embeddings import OpenAIEmbeddings
                    cls._instance = OpenAIEmbeddings(
                        openai_api_key=settings.OPENAI_API_KEY,
                        model="text-embedding-3-small"
                    )
                    cls._provider_used = "openai"
                    cls._model_name = "text-embedding-3-small"
                    return cls._instance
                except Exception as e:
                    logger.warning(f"Erreur OpenAI embeddings ({e}). Basculement vers local.")

        # Default fallback
        logger.info("Utilisation des embeddings légers locaux (LightweightLocalEmbeddings).")
        cls._instance = LightweightLocalEmbeddings(dimension=384)
        cls._provider_used = "lightweight-local"
        cls._model_name = "hashing-384"
        return cls._instance

    @classmethod
    def get_info(cls) -> dict:
        return {
            "provider": cls._provider_used,
            "model_name": cls._model_name
        }
