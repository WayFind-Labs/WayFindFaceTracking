from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from biometrics.base import EmbeddingVector
from biometrics.modality import BiometricModality


@dataclass
class MatchResultItem:
    """Detailed comparison result for a single candidate match."""
    candidate_embedding_id: str
    person_id: Optional[str]
    similarity_score: float
    cosine_distance: float
    euclidean_distance: float
    is_match: bool
    confidence_percent: float
    modality: BiometricModality
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_embedding_id": self.candidate_embedding_id,
            "person_id": self.person_id,
            "similarity_score": round(self.similarity_score, 4),
            "cosine_distance": round(self.cosine_distance, 4),
            "euclidean_distance": round(self.euclidean_distance, 4),
            "is_match": self.is_match,
            "confidence_percent": round(self.confidence_percent, 2),
            "modality": self.modality.value,
            "metadata": self.metadata,
        }


class BiometricMatchEngine:
    """
    Mathematical similarity matching engine for multi-modal biometric vectors.
    Enforces strict modality boundaries and calculates cosine and Euclidean distance metrics.
    """

    # Default similarity thresholds per modality (cosine similarity scale 0.0 to 1.0)
    DEFAULT_THRESHOLDS = {
        BiometricModality.FACE: 0.50,
        BiometricModality.FINGERPRINT: 0.60,
        BiometricModality.BODY_MARK: 0.55,
        BiometricModality.IRIS: 0.70,
    }

    @classmethod
    def compare_two(
        cls,
        vector_a: EmbeddingVector,
        vector_b: EmbeddingVector,
        threshold: Optional[float] = None,
    ) -> MatchResultItem:
        """
        Compare two biometric vectors and evaluate similarity.
        Enforces strict modality equality.
        """
        if vector_a.modality != vector_b.modality:
            raise ValueError(
                f"MODALITY_MISMATCH: Cannot compare '{vector_a.modality.value}' "
                f"embedding with '{vector_b.modality.value}' embedding."
            )

        if vector_a.dimension != vector_b.dimension:
            raise ValueError(
                f"DIMENSION_MISMATCH: Embedding vector dimensions do not match "
                f"({vector_a.dimension} vs {vector_b.dimension})."
            )

        # Apply default modality threshold if none supplied
        if threshold is None:
            threshold = cls.DEFAULT_THRESHOLDS.get(vector_a.modality, 0.50)

        v1 = vector_a.vector
        v2 = vector_b.vector

        # Cosine similarity (dot product of normalized vectors)
        cos_sim = float(np.dot(v1, v2))
        cos_dist = float(1.0 - cos_sim)
        euc_dist = float(np.linalg.norm(v1 - v2))

        is_match = bool(cos_sim >= threshold)
        confidence = float(np.clip(cos_sim * 100.0, 0.0, 100.0))

        return MatchResultItem(
            candidate_embedding_id=vector_b.embedding_id,
            person_id=vector_b.person_id,
            similarity_score=cos_sim,
            cosine_distance=cos_dist,
            euclidean_distance=euc_dist,
            is_match=is_match,
            confidence_percent=confidence,
            modality=vector_a.modality,
            metadata=vector_b.metadata,
        )

    @classmethod
    def match_candidates(
        cls,
        query_vector: EmbeddingVector,
        candidate_vectors: List[EmbeddingVector],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[MatchResultItem]:
        """
        Rank candidate vectors against a query vector by descending similarity score.
        Filters out any candidate that fails modality validation.
        """
        results: List[MatchResultItem] = []

        for candidate in candidate_vectors:
            if candidate.modality != query_vector.modality:
                continue
            item = cls.compare_two(query_vector, candidate, threshold=threshold)
            results.append(item)

        # Sort descending by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        if top_k is not None and top_k > 0:
            return results[:top_k]
        return results

    @classmethod
    def fuse_scores(
        cls,
        scores_by_modality: Dict[BiometricModality, float],
        custom_weights: Optional[Dict[BiometricModality, float]] = None,
    ) -> float:
        """
        Calculate weighted fusion score across multiple biometric modalities.
        Weights are normalized automatically across present modalities.
        """
        if not scores_by_modality:
            return 0.0

        default_weights = {
            BiometricModality.FACE: 0.50,
            BiometricModality.FINGERPRINT: 0.30,
            BiometricModality.BODY_MARK: 0.20,
            BiometricModality.IRIS: 0.50,
        }
        weights = custom_weights or default_weights

        total_weight = 0.0
        weighted_sum = 0.0

        for modality, score in scores_by_modality.items():
            w = weights.get(modality, 0.25)
            weighted_sum += w * float(np.clip(score, 0.0, 1.0))
            total_weight += w

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 4)
