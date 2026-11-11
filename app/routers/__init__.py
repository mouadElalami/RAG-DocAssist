from .documents import router as documents_router
from .search import router as search_router
from .chat import router as chat_router
from .config_routes import router as config_router

__all__ = ["documents_router", "search_router", "chat_router", "config_router"]
