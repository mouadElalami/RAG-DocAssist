"""
Lanceur de la plateforme RAG-DocAssist
"""
import sys
import uvicorn
from app.config import settings

def main():
    print("=" * 65)
    print("🚀 Démarrage de RAG-DocAssist — Plateforme Web IA Générative & API")
    print("=" * 65)
    print(f"📍 Interface Web Réactive : http://{settings.HOST}:{settings.PORT}")
    print(f"📖 Documentation Swagger  : http://{settings.HOST}:{settings.PORT}/docs")
    print(f"📚 Documentation ReDoc    : http://{settings.HOST}:{settings.PORT}/redoc")
    print(f"🧠 Index FAISS            : {settings.FAISS_INDEX_DIR}")
    print(f"🤖 Fournisseur LLM actif  : {settings.LLM_PROVIDER}")
    print("=" * 65)

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    main()
