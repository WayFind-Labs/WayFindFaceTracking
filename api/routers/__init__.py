from api.routers.embed import router as embed_router
from api.routers.health import router as health_router
from api.routers.images import router as images_router
from api.routers.match import router as match_router

__all__ = [
    "health_router",
    "embed_router",
    "match_router",
    "images_router",
]
