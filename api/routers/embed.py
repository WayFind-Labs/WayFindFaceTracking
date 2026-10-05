import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.dependencies import get_registry, get_store
from api.schemas.embed import DirectVectorEmbedRequest, EmbedResponse
from biometrics.base import EmbeddingVector
from biometrics.modality import BiometricModality
from biometrics.registry import ExtractorRegistry
from biometrics.store import EmbeddingStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/biometrics", tags=["Biometric Embeddings"])


@router.post(
    "/embed",
    response_model=EmbedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Extract & persist a biometric embedding from an uploaded image",
)
async def extract_and_store_embedding(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    modality: str = Form(
        default="FACE", 
        description="Biometric modality: FACE, FINGERPRINT, BODY_MARK, IRIS"
    ),
    person_id: Optional[str] = Form(
        default=None, 
        description="Person / Child UUID if registering a known identity"
    ),
    metadata: Optional[str] = Form(
        default=None, 
        description="Optional JSON string of custom metadata key-values"
    ),
    registry: ExtractorRegistry = Depends(get_registry),
    store: EmbeddingStore = Depends(get_store),
) -> EmbedResponse:
    """
    Extracts a normalized biometric feature vector from an uploaded image using the
    registered Strategy for the specified modality, caches it in the EmbeddingStore,
    and returns the unique `embedding_id`.
    """
    # 1. Parse and validate modality
    try:
        bio_modality = BiometricModality.from_str(modality)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # 2. Check extractor availability
    if not registry.is_supported(bio_modality):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Modality '{bio_modality.value}' is registered as a roadmap stub "
                f"but feature extraction is not yet operational."
            ),
        )

    # 3. Parse optional metadata JSON
    meta_dict = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
            if not isinstance(meta_dict, dict):
                raise ValueError("Metadata must be a JSON object.")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata JSON string: {exc}",
            )

    # 4. Read image bytes
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Uploaded image file is empty (0 bytes).")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded image bytes: {exc}",
        )

    # 5. Extract embedding via strategy
    extractor = registry.get(bio_modality)
    try:
        embedding = extractor.extract(
            image_input=image_bytes,
            person_id=person_id,
            metadata=meta_dict,
        )
    except ValueError as exc:
        # e.g., No face detected in image
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Biometric extraction failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during biometric extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal extraction error: {exc}",
        )

    # 6. Save in store
    store.save(embedding)

    return EmbedResponse(
        embedding_id=embedding.embedding_id,
        modality=embedding.modality.value,
        dimension=embedding.dimension,
        model_version=embedding.model_version,
        confidence=embedding.confidence,
        person_id=embedding.person_id,
        metadata=embedding.metadata,
        created_at=embedding.created_at,
    )


@router.post(
    "/embed/direct",
    response_model=EmbedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Directly register a pre-computed normalized vector",
)
async def direct_vector_embed(
    payload: DirectVectorEmbedRequest,
    store: EmbeddingStore = Depends(get_store),
) -> EmbedResponse:
    """
    Directly register a pre-computed vector into the store without re-running neural extraction.
    Useful for testing, syncing across distributed nodes, or importing legacy embeddings.
    """
    try:
        bio_modality = BiometricModality.from_str(payload.modality)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    embedding = EmbeddingVector.create(
        modality=bio_modality,
        vector=payload.vector,
        model_version=payload.model_version,
        confidence=1.0,
        person_id=payload.person_id,
        metadata=payload.metadata or {},
    )

    store.save(embedding)

    return EmbedResponse(
        embedding_id=embedding.embedding_id,
        modality=embedding.modality.value,
        dimension=embedding.dimension,
        model_version=embedding.model_version,
        confidence=embedding.confidence,
        person_id=embedding.person_id,
        metadata=embedding.metadata,
        created_at=embedding.created_at,
    )
