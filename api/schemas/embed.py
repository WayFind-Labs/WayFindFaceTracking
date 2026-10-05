from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmbedResponse(BaseModel):
    """API Response returned after successful biometric embedding extraction."""
    embedding_id: str = Field(..., description="Unique UUID identifying this generated embedding")
    modality: str = Field(..., description="Biometric modality (FACE, FINGERPRINT, etc.)")
    dimension: int = Field(..., description="Embedding vector length (e.g. 512 for ArcFace)")
    model_version: str = Field(..., description="Underlying neural model and version")
    confidence: float = Field(..., description="Extraction confidence score (0.0 to 1.0)")
    person_id: Optional[str] = Field(None, description="Associated Person/Child UUID if provided")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp")


class DirectVectorEmbedRequest(BaseModel):
    """Payload to directly register a pre-extracted raw vector."""
    vector: List[float] = Field(..., description="List of float values representing normalized embedding")
    modality: str = Field(default="FACE", description="Biometric modality")
    model_version: str = Field(default="custom-v1", description="Model identifier")
    person_id: Optional[str] = Field(None, description="Associated Person/Child UUID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
