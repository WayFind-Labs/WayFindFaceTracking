import os
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

from biometrics.base import BiometricExtractor, EmbeddingVector
from biometrics.modality import BiometricModality


class ArcFaceExtractor(BiometricExtractor):
    """
    ArcFace (InsightFace) face feature extractor implementing BiometricExtractor strategy.
    Produces normalized 512-dimensional face embeddings.
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        providers: Optional[list] = None,
        lazy_load: bool = True,
    ):
        self._model_name = model_name
        self._providers = providers or ["CPUExecutionProvider"]
        self._service = None
        self._lazy_load = lazy_load
        
        if not lazy_load:
            self._init_service()

    def _init_service(self):
        if self._service is None:
            from arcface_service import ArcFaceService
            self._service = ArcFaceService(
                model_name=self._model_name,
                providers=self._providers,
            )

    @property
    def modality(self) -> BiometricModality:
        return BiometricModality.FACE

    @property
    def model_version(self) -> str:
        return f"arcface-{self._model_name}-v1"

    @property
    def dimension(self) -> int:
        return 512

    @property
    def is_available(self) -> bool:
        return True

    def extract(
        self,
        image_input: Union[str, np.ndarray, bytes],
        person_id: Optional[str] = None,
        embedding_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingVector:
        """
        Extract 512-D face embedding from image path, numpy array, or raw bytes.
        """
        self._init_service()

        # Handle raw bytes input (e.g. from FastAPI file upload)
        if isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from provided byte stream.")
            processed_input = img
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
            processed_input = img
        elif isinstance(image_input, np.ndarray):
            processed_input = image_input
        else:
            raise TypeError("image_input must be file path, numpy ndarray, or bytes.")

        # Extract using ArcFaceService
        raw_vector = self._service.image_to_vector(processed_input)

        return EmbeddingVector.create(
            modality=self.modality,
            vector=raw_vector,
            model_version=self.model_version,
            confidence=0.98,
            person_id=person_id,
            embedding_id=embedding_id,
            metadata=metadata or {},
        )
