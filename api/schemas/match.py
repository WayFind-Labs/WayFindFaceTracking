from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. Canonical Biometric Matching Schemas
# ==========================================

class BiometricCandidateItem(BaseModel):
    """Detailed similarity metrics for a candidate match."""
    candidate_embedding_id: str
    person_id: Optional[str] = None
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    cosine_distance: float = Field(..., description="1 - cosine similarity")
    euclidean_distance: float = Field(..., description="L2 Euclidean distance")
    is_match: bool = Field(..., description="True if similarity_score >= threshold")
    confidence_percent: float = Field(..., description="Similarity expressed as 0-100%")
    modality: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BiometricMatchRequest(BaseModel):
    """Request payload for comparing a query embedding against candidate embeddings."""
    query_embedding_id: str = Field(..., description="UUID of the query embedding")
    candidate_embedding_ids: Optional[List[str]] = Field(
        None, 
        description="Specific candidate embedding IDs to search. If omitted, searches all embeddings of matching modality in store."
    )
    modality: str = Field(default="FACE", description="Biometric modality to match")
    threshold: Optional[float] = Field(
        None, 
        description="Cosine similarity cutoff threshold (defaults: FACE=0.50, FINGERPRINT=0.60)"
    )
    top_k: Optional[int] = Field(
        5, 
        description="Maximum number of top matches to return"
    )


class BiometricMatchResponse(BaseModel):
    """Response payload containing ranked candidate matches."""
    matches: List[BiometricCandidateItem]
    query_embedding_id: str
    modality: str
    candidates_searched: int
    top_k: int


# ==========================================
# 2. WayFind MCP Client Shim Schemas
# (Matches ImageBackendClient & schemas/image.py)
# ==========================================

class ImageCompareRequest(BaseModel):
    """
    Contract consumed by WayFindMCPClient ImageBackendClient:
    POST /api/images/compare { media_id: str, candidate_case_ids: list[str] }
    """
    media_id: str = Field(..., description="Media / Embedding ID of the query image")
    candidate_case_ids: List[str] = Field(..., description="List of candidate case IDs / embedding IDs")


class ImageCompareMatch(BaseModel):
    """Single candidate match format expected by WayFindMCPClient."""
    candidate_case_id: str
    similarity_score: float
    model_version: str = "arcface-buffalo_sc-v1"
    match_regions: List[str] = Field(
        default_factory=lambda: ["facial structure", "facial landmarks", "feature vector"]
    )


class ImageCompareResponse(BaseModel):
    """Response format expected by WayFindMCPClient ImageBackendClient."""
    matches: List[ImageCompareMatch]
    query_media_id: str
    candidates_searched: int
