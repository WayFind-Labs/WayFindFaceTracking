from contextlib import asynccontextmanager
import logging
import os
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import embed_router, health_router, images_router, match_router
from biometrics.registry import get_extractor_registry

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wayfind-facetracking")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    registry = get_extractor_registry()
    modalities = [m["modality"] for m in registry.list_supported_modalities()]
    logger.info("==========================================================")
    logger.info("  WayFind Biometric Vision Microservice Starting (Port 8003)")
    logger.info("  Registered Modalities: %s", ", ".join(modalities))
    logger.info("==========================================================")
    yield
    logger.info("WayFind Biometric Vision Microservice shutting down...")


app = FastAPI(
    title="WayFind Biometric Vision Microservice",
    description=(
        "Standalone Multi-Modal Biometric Extraction, Storage, and Vector Matching Engine "
        "for the WayFind Intelligent Missing Child Recovery Platform."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Next.js PWA Client and FastAPI Backends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    """Translate domain validation and modality mismatch errors into structured HTTP 400."""
    error_msg = str(exc)
    status_code = (
        status.HTTP_400_BAD_REQUEST 
        if "MODALITY_MISMATCH" in error_msg or "DIMENSION_MISMATCH" in error_msg 
        else status.HTTP_422_UNPROCESSABLE_ENTITY
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": error_msg, "error_type": "ValidationError"},
    )


# Mount routers
app.include_router(health_router)
app.include_router(embed_router)
app.include_router(match_router)
app.include_router(images_router)


@app.get("/", tags=["Root"])
async def root_status():
    """Service landing endpoint."""
    return {
        "service": "WayFind Biometric Vision Microservice",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8003"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
