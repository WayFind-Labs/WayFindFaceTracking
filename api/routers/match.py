import logging
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_store
from api.schemas.match import (
    BiometricCandidateItem,
    BiometricMatchRequest,
    BiometricMatchResponse,
)
from biometrics.match_engine import BiometricMatchEngine
from biometrics.modality import BiometricModality
from biometrics.store import EmbeddingStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/biometrics", tags=["Biometric Matching"])


@router.post(
    "/match",
    response_model=BiometricMatchResponse,
    summary="Match a query biometric vector against candidate vectors with modality safety",
)
async def match_candidates(
    payload: BiometricMatchRequest,
    store: EmbeddingStore = Depends(get_store),
) -> BiometricMatchResponse:
    """
    Compares a stored query embedding against a list of candidate embedding IDs.
    Enforces strict modality boundaries and calculates cosine similarity, distance, and match verdicts.
    """
    # 1. Validate modality
    try:
        bio_modality = BiometricModality.from_str(payload.modality)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # 2. Retrieve query embedding
    query_emb = store.get(payload.query_embedding_id)
    if not query_emb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query embedding ID '{payload.query_embedding_id}' not found in store.",
        )

    # 3. Check query modality matches requested modality
    if query_emb.modality != bio_modality:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"MODALITY_MISMATCH: Query embedding is of modality '{query_emb.modality.value}', "
                f"but request requested match against '{bio_modality.value}'."
            ),
        )

    # 4. Gather candidate embeddings
    if payload.candidate_embedding_ids is not None:
        # User specified explicit candidate IDs (e.g. top shortlisted cases)
        candidate_embs = store.get_many(payload.candidate_embedding_ids)
    else:
        # Search all embeddings of this modality in the store
        # Exclude query itself
        candidate_embs = [
            store.get(eid)
            for eid in store._by_modality.get(bio_modality, set())
            if eid != payload.query_embedding_id and store.get(eid) is not None
        ]

    if not candidate_embs:
        return BiometricMatchResponse(
            matches=[],
            query_embedding_id=payload.query_embedding_id,
            modality=bio_modality.value,
            candidates_searched=0,
            top_k=payload.top_k or 5,
        )

    # 5. Execute ranking and matching
    try:
        ranked_items = BiometricMatchEngine.match_candidates(
            query_vector=query_emb,
            candidate_vectors=candidate_embs,
            threshold=payload.threshold,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # 6. Build response
    match_items = [
        BiometricCandidateItem(
            candidate_embedding_id=item.candidate_embedding_id,
            person_id=item.person_id,
            similarity_score=item.similarity_score,
            cosine_distance=item.cosine_distance,
            euclidean_distance=item.euclidean_distance,
            is_match=item.is_match,
            confidence_percent=item.confidence_percent,
            modality=item.modality.value,
            metadata=item.metadata,
        )
        for item in ranked_items
    ]

    return BiometricMatchResponse(
        matches=match_items,
        query_embedding_id=payload.query_embedding_id,
        modality=bio_modality.value,
        candidates_searched=len(candidate_embs),
        top_k=payload.top_k or 5,
    )
