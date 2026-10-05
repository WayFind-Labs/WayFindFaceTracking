from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import uuid
import numpy as np

from biometrics.modality import BiometricModality


@dataclass
class EmbeddingVector:
    """
    Unified multi-modal biometric embedding envelope.
    All modalities (Face, Fingerprint, Body Mark, Iris) produce this exact structure.
    """
    embedding_id: str
    modality: BiometricModality
    vector: np.ndarray
    dimension: int
    model_version: str
    confidence: float
    person_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        """Convert embedding envelope to dictionary for serialization."""
        data = {
            "embedding_id": self.embedding_id,
            "modality": self.modality.value,
            "dimension": self.dimension,
            "model_version": self.model_version,
            "confidence": round(float(self.confidence), 4),
            "person_id": self.person_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
        if include_vector:
            data["vector"] = self.vector.tolist()
        return data

    @classmethod
    def create(
        cls,
        modality: BiometricModality,
        vector: Union[np.ndarray, List[float]],
        model_version: str,
        confidence: float = 1.0,
        person_id: Optional[str] = None,
        embedding_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EmbeddingVector":
        """Factory method ensuring normalized 1D float32 numpy vector."""
        if isinstance(vector, list):
            np_vec = np.array(vector, dtype=np.float32)
        elif isinstance(vector, np.ndarray):
            np_vec = vector.astype(np.float32)
        else:
            raise TypeError("Vector must be a list of floats or numpy ndarray.")

        np_vec = np_vec.flatten()
        norm = np.linalg.norm(np_vec)
        if norm > 0:
            np_vec = np_vec / norm

        return cls(
            embedding_id=embedding_id or str(uuid.uuid4()),
            modality=modality,
            vector=np_vec,
            dimension=len(np_vec),
            model_version=model_version,
            confidence=max(0.0, min(1.0, float(confidence))),
            person_id=person_id,
            metadata=metadata or {},
        )


class BiometricExtractor(ABC):
    """
    Abstract Strategy for extracting biometric embeddings from image/sensor data.
    Each modality implements this interface.
    """

    @property
    @abstractmethod
    def modality(self) -> BiometricModality:
        """The modality handled by this extractor."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Name and version of the underlying feature extraction model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension produced by this extractor."""
        pass

    @property
    def is_available(self) -> bool:
        """Whether this extractor is fully loaded and operational."""
        return True

    @abstractmethod
    def extract(
        self,
        image_input: Union[str, np.ndarray, bytes],
        person_id: Optional[str] = None,
        embedding_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingVector:
        """
        Extract normalized embedding vector from image/sensor input.

        Args:
            image_input: File path, numpy image array (BGR/RGB), or raw image bytes.
            person_id: Optional ID of the person being registered.
            embedding_id: Optional pre-assigned embedding UUID.
            metadata: Optional additional key-value metadata.

        Returns:
            EmbeddingVector envelope.
        """
        pass
