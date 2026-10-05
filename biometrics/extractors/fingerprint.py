from typing import Any, Dict, Optional, Union
import numpy as np

from biometrics.base import BiometricExtractor, EmbeddingVector
from biometrics.modality import BiometricModality


class FingerprintExtractor(BiometricExtractor):
    """
    Fingerprint feature extractor strategy (Roadmap stub).
    Will be implemented with a deep fingerprint/minutiae embedding neural network.
    """

    def __init__(self, model_version: str = "fingerprint-deep-v1"):
        self._model_version = model_version

    @property
    def modality(self) -> BiometricModality:
        return BiometricModality.FINGERPRINT

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return 256

    @property
    def is_available(self) -> bool:
        return False

    def extract(
        self,
        image_input: Union[str, np.ndarray, bytes],
        person_id: Optional[str] = None,
        embedding_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingVector:
        raise NotImplementedError(
            "Fingerprint extraction is scheduled for Phase 4 of the WayFind roadmap. "
            "Model weights and minutiae pipeline are not yet loaded."
        )
