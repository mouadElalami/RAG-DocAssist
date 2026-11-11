import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.routers import documents_router, search_router, chat_router, config_router

description = """
# 🚀 RAG-DocAssist — Plateforme Web IA Générative & API REST

Bienvenue sur l'API de **RAG-DocAssist**, une solution documentaire de pointe articulant :
- 📄 **Ingestion & Vectorisation documentaire** (PDF, DOCX, TXT, Markdown) avec découpage intelligent (*Chunking*).
- 🧠 **Base vectorielle FAISS** (*Facebook AI Similarity Search*) pour une recherche sémantique ultra-rapide par vecteurs d'embeddings.
- 💬 **Génération augmentée par récupération (RAG)** connectée aux LLMs de référence (*Google Gemini, OpenAI GPT, Ollama*) avec citations des sources.
- 🌐 **Interface Web Réactive** moderne et interactive.
"""

app = FastAPI(
    title=settings.APP_NAME,
    description=description,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Documents & Vectorisation",
            "description": "Gestion des fichiers, extraction textuelle, découpage vectoriel et indexation FAISS."
        },
        {
            "name": "Recherche Sémantique",
            "description": "Recherche vectorielle directe par similarité cosinus / distance L2."
        },
        {
            "name": "Question / Réponse RAG",
            "description": "Pipeline RAG complet générant des réponses contextuelles sourcées avec un LLM."
        },
        {
            "name": "Système & Configuration",
            "description": "Statuts de l'index, paramètres de modèles et vérification d'intégrité."
        }
    ]
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(config_router)

# Mount Static Files
STATIC_DIR = settings.BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", summary="Interface Web RAG-DocAssist", include_in_schema=False)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Bienvenue sur RAG-DocAssist API. L'interface web est en cours d'initialisation.",
        "documentation": "/docs"
    }
