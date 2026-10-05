from functools import lru_cache
import logging
from typing import Any, Dict, List, Optional

from biometrics.base import BiometricExtractor
from biometrics.extractors.body_mark import BodyMarkExtractor
from biometrics.extractors.face import ArcFaceExtractor
from biometrics.extractors.fingerprint import FingerprintExtractor
from biometrics.modality import BiometricModality

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """
    Central Registry for BiometricExtractor strategies.
    Allows zero-code-change registration of new biometric modalities.
    """

    def __init__(self):
        self._extractors: Dict[BiometricModality, BiometricExtractor] = {}

    def register(self, extractor: BiometricExtractor, override: bool = True) -> None:
        """Register an extractor strategy for its modality."""
        modality = extractor.modality
        if modality in self._extractors and not override:
            raise ValueError(f"Extractor already registered for modality '{modality.value}'.")
        self._extractors[modality] = extractor
        logger.info(
            "Registered biometric extractor for modality '%s' (model: %s, dim: %d)",
            modality.value,
            extractor.model_version,
            extractor.dimension,
        )

    def get(self, modality: BiometricModality) -> BiometricExtractor:
        """Get the extractor strategy for a given modality."""
        if modality not in self._extractors:
            raise KeyError(
                f"No biometric extractor registered for modality '{modality.value}'. "
                f"Registered modalities: {[m.value for m in self._extractors.keys()]}"
            )
        return self._extractors[modality]

    def is_supported(self, modality: BiometricModality) -> bool:
        """Check if an extractor is registered and available for a modality."""
        if modality not in self._extractors:
            return False
        return self._extractors[modality].is_available

    def list_supported_modalities(self) -> List[Dict[str, Any]]:
        """List all registered modalities with their readiness status and model info."""
        result = []
        for modality, extractor in self._extractors.items():
            result.append({
                "modality": modality.value,
                "is_available": extractor.is_available,
                "model_version": extractor.model_version,
                "dimension": extractor.dimension,
            })
        return result


def create_default_registry() -> ExtractorRegistry:
    """Instantiate and pre-register default extractors for all modalities."""
    registry = ExtractorRegistry()
    
    # Active: Face (ArcFace)
    registry.register(ArcFaceExtractor(model_name="buffalo_sc", lazy_load=True))
    
    # Stubs: Fingerprint & Body Mark
    registry.register(FingerprintExtractor())
    registry.register(BodyMarkExtractor())
    
    return registry


@lru_cache(maxsize=1)
def get_extractor_registry() -> ExtractorRegistry:
    """Singleton getter for the default ExtractorRegistry."""
    return create_default_registry()
