from fastapi import APIRouter, Depends
from typing import Any, Dict

from api.dependencies import get_registry, get_store
from biometrics.registry import ExtractorRegistry
from biometrics.store import EmbeddingStore

router = APIRouter(tags=["Health & Status"])


@router.get("/health", summary="Service health and biometric readiness")
async def health_check(
    registry: ExtractorRegistry = Depends(get_registry),
    store: EmbeddingStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Returns service health status, registered modalities, and current stored embedding count.
    """
    return {
        "status": "ok",
        "service": "WayFindFaceTracking",
        "version": "1.0.0",
        "supported_modalities": registry.list_supported_modalities(),
        "total_cached_embeddings": store.count(),
    }
