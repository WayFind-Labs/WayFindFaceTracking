from collections import defaultdict
from functools import lru_cache
import logging
import threading
from typing import Dict, List, Optional, Set

from biometrics.base import EmbeddingVector
from biometrics.modality import BiometricModality

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Thread-safe in-memory store for multi-modal biometric embeddings.
    Provides fast lookup by embedding ID, person ID, and modality indexing.
    Can be seamlessly swapped with Redis or a Vector DB (Qdrant/Milvus) in Phase 4+.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._embeddings: Dict[str, EmbeddingVector] = {}
        self._by_person: Dict[str, Set[str]] = defaultdict(set)
        self._by_modality: Dict[BiometricModality, Set[str]] = defaultdict(set)

    def save(self, embedding: EmbeddingVector) -> str:
        """Store an embedding vector and update secondary indices."""
        with self._lock:
            eid = embedding.embedding_id
            self._embeddings[eid] = embedding
            self._by_modality[embedding.modality].add(eid)
            
            if embedding.person_id:
                self._by_person[embedding.person_id].add(eid)

            logger.debug(
                "Stored embedding %s for modality %s (person_id: %s)",
                eid,
                embedding.modality.value,
                embedding.person_id,
            )
            return eid

    def get(self, embedding_id: str) -> Optional[EmbeddingVector]:
        """Retrieve an embedding by ID."""
        with self._lock:
            return self._embeddings.get(embedding_id)

    def get_many(self, embedding_ids: List[str]) -> List[EmbeddingVector]:
        """Retrieve multiple embeddings by IDs in requested order (skips missing)."""
        with self._lock:
            results = []
            for eid in embedding_ids:
                if eid in self._embeddings:
                    results.append(self._embeddings[eid])
            return results

    def get_by_person(
        self, person_id: str, modality: Optional[BiometricModality] = None
    ) -> List[EmbeddingVector]:
        """Retrieve all embeddings associated with a person ID, optionally filtered by modality."""
        with self._lock:
            if person_id not in self._by_person:
                return []
            
            eids = self._by_person[person_id]
            results = []
            for eid in eids:
                emb = self._embeddings.get(eid)
                if emb:
                    if modality is None or emb.modality == modality:
                        results.append(emb)
            return results

    def delete(self, embedding_id: str) -> bool:
        """Remove an embedding from the store and purge all indices."""
        with self._lock:
            emb = self._embeddings.pop(embedding_id, None)
            if not emb:
                return False

            self._by_modality[emb.modality].discard(embedding_id)
            if emb.person_id and emb.person_id in self._by_person:
                self._by_person[emb.person_id].discard(embedding_id)
                if not self._by_person[emb.person_id]:
                    del self._by_person[emb.person_id]
            return True

    def count(self, modality: Optional[BiometricModality] = None) -> int:
        """Count stored embeddings, optionally filtered by modality."""
        with self._lock:
            if modality is None:
                return len(self._embeddings)
            return len(self._by_modality.get(modality, set()))

    def clear(self) -> None:
        """Clear all stored embeddings and indices."""
        with self._lock:
            self._embeddings.clear()
            self._by_person.clear()
            self._by_modality.clear()


@lru_cache(maxsize=1)
def get_embedding_store() -> EmbeddingStore:
    """Singleton getter for the global in-memory EmbeddingStore."""
    return EmbeddingStore()
