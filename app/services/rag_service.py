import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

from app.config import settings
from app.models.schemas import ChatMessage, Citation, ChatResponse
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    def ask(
        self,
        query: str,
        top_k: int = 4,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        history: Optional[List[ChatMessage]] = None
    ) -> ChatResponse:
        start_total = time.perf_counter()

        # Step 1: Retrieval
        start_retrieval = time.perf_counter()
        retrieved_docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        retrieval_time_ms = round((time.perf_counter() - start_retrieval) * 1000, 2)

        # Step 2: Format Citations
        citations: List[Citation] = []
        context_parts: List[str] = []

        for idx, (doc, score) in enumerate(retrieved_docs_with_scores, start=1):
            source = doc.metadata.get("source", "Inconnu")
            doc_id = doc.metadata.get("doc_id", "doc")
            chunk_index = doc.metadata.get("chunk_index", 0)
            page = doc.metadata.get("page")
            
            # Distance to score normalization (FAISS L2 distance: lower is better)
            # Normalizing to a 0..1 relevance indicator
            rel_score = round(1.0 / (1.0 + score), 4)

            citations.append(
                Citation(
                    citation_id=idx,
                    source=source,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    page=page,
                    relevance_score=rel_score,
                    excerpt=doc.page_content[:250].strip() + ("..." if len(doc.page_content) > 250 else "")
                )
            )

            page_info = f" (Page {page})" if page else ""
            context_parts.append(
                f"[Source {idx}: {source}{page_info} | Chunk {chunk_index}]\n{doc.page_content}"
            )

        context_text = "\n\n".join(context_parts) if context_parts else "Aucun document pertinent trouvé dans la base vectorielle."

        # Step 3: LLM Provider selection & Generation
        active_provider = (provider or settings.LLM_PROVIDER).lower()
        active_api_key = api_key or (settings.GEMINI_API_KEY if active_provider == "gemini" else settings.OPENAI_API_KEY)

        start_gen = time.perf_counter()
        answer, model_used = self._generate_answer(
            query=query,
            context=context_text,
            has_context=bool(retrieved_docs_with_scores),
            provider=active_provider,
            api_key=active_api_key,
            temperature=temperature,
            custom_system_prompt=system_prompt,
            history=history or []
        )
        generation_time_ms = round((time.perf_counter() - start_gen) * 1000, 2)
        total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)

        return ChatResponse(
            query=query,
            answer=answer,
            sources=citations,
            provider=active_provider,
            model=model_used,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms
        )

    def _generate_answer(
        self,
        query: str,
        context: str,
        has_context: bool,
        provider: str,
        api_key: Optional[str],
        temperature: float,
        custom_system_prompt: Optional[str],
        history: List[ChatMessage]
    ) -> Tuple[str, str]:
        if not has_context:
            return (
                "⚠️ Aucun document n'est indexé dans la base vectorielle. "
                "Veuillez d'abord téléverser ou indexer des documents (PDF, DOCX, TXT, Markdown) pour permettre la recherche sémantique.",
                "none"
            )

        default_system = (
            "Tu es un assistant IA expert et rigoureux spécialisé dans l'analyse documentaire (RAG-DocAssist).\n"
            "Réponds à la question de l'utilisateur de manière synthétique, claire et structurée (en français sauf si la question est dans une autre langue).\n"
            "CONSIGNES STRICTES :\n"
            "1. Base ta réponse EXCLUSIVEMENT sur les extraits de documents fournis dans le contexte.\n"
            "2. Pour chaque fait ou affirmation clé, indique la citation correspondante sous la forme [Source 1], [Source 2], etc.\n"
            "3. Si l'information ne se trouve pas dans les extraits, affirme-le en toute transparence sans rien inventer.\n"
            "4. Utilise la mise en forme Markdown (titres, listes à puces, gras) pour une lecture agréable."
        )

        sys_prompt = custom_system_prompt or default_system

        # Build prompt
        prompt = f"{sys_prompt}\n\n=== DOCUMENTS DE CONTEXTE ===\n{context}\n\n"
        if history:
            prompt += "=== HISTORIQUE DE CONVERSATION ===\n"
            for msg in history[-4:]:  # Keep last 4 turns
                prompt += f"{msg.role.capitalize()}: {msg.content}\n"
            prompt += "\n"

        prompt += f"Question de l'utilisateur : {query}\n\nRéponse :"

        # Route by provider
        if provider == "gemini" and api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_name = settings.GEMINI_MODEL or "gemini-1.5-flash"
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                return resp.text, model_name
            except Exception as e:
                logger.error(f"Erreur Gemini: {e}")
                return self._mock_generation(query, context, f"Erreur Gemini ({e})")

        elif provider == "openai" and api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                model_name = settings.OPENAI_MODEL or "gpt-4o-mini"
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"}
                    ],
                    temperature=temperature
                )
                return resp.choices[0].message.content, model_name
            except Exception as e:
                logger.error(f"Erreur OpenAI: {e}")
                return self._mock_generation(query, context, f"Erreur OpenAI ({e})")

        elif provider == "ollama":
            try:
                import httpx
                model_name = settings.OLLAMA_MODEL or "llama3"
                url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
                res = httpx.post(url, json=payload, timeout=60.0)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("response", ""), model_name
                else:
                    return self._mock_generation(query, context, f"Ollama HTTP {res.status_code}")
            except Exception as e:
                logger.error(f"Erreur Ollama: {e}")
                return self._mock_generation(query, context, f"Erreur Ollama ({e})")

        # Fallback / Offline / Mock Mode
        return self._mock_generation(query, context)

    def _mock_generation(self, query: str, context: str, note: Optional[str] = None) -> Tuple[str, str]:
        """
        Synthesizes an answer based directly on the retrieved context chunks,
        ensuring full offline capability and testing without external API keys.
        """
        lines = [c.strip() for c in context.split("\n") if c.strip() and not c.startswith("===")]
        
        # Extract snippets from context chunks
        snippets = []
        for line in lines:
            if line.startswith("[Source"):
                snippets.append(f"\n**{line}** :")
            elif len(line) > 20:
                snippets.append(f"- {line[:160]}...")
            if len(snippets) >= 6:
                break

        summary = "\n".join(snippets)

        note_header = f"> ℹ️ *Mode Démo Local / Hors-ligne (ou clef API absente).* Configurez une clef Gemini ou OpenAI dans les paramètres pour un LLM distant.\n\n" if not note else f"> ⚠️ *{note}* (Basculement en mode d'extraction contextuelle)\n\n"

        answer = (
            f"{note_header}"
            f"### Synthèse documentaire pour la question : *« {query} »*\n\n"
            f"D'après les documents indexés dans la base sémantique FAISS, voici les informations pertinentes extraites :\n\n"
            f"{summary}\n\n"
            f"Ces extraits correspondent aux passages les plus proches sémantiquement de votre requête avec citations des sources."
        )

        return answer, "rag-local-synthesizer"
