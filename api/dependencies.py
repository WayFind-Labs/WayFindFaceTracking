from biometrics.registry import ExtractorRegistry, get_extractor_registry
from biometrics.store import EmbeddingStore, get_embedding_store


def get_registry() -> ExtractorRegistry:
    """Dependency provider for the ExtractorRegistry singleton."""
    return get_extractor_registry()


def get_store() -> EmbeddingStore:
    """Dependency provider for the EmbeddingStore singleton."""
    return get_embedding_store()
