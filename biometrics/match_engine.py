from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
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


@dataclass
class FusedMatchResult:
    """Structured result for weighted multi-biometric score fusion."""
    fused_score: float  # Percentage scale: 0.0 to 100.0%
    fused_score_raw: float  # Scale: 0.0 to 1.0
    individual_scores: Dict[str, float]  # Modality to individual cosine score
    is_match: bool  # True if fused_score >= threshold_percent
    threshold_percent: float  # Unified match threshold percentage (default 40.0%)
    weights_used: Dict[str, float]  # Normalized weight applied per modality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fused_score": round(self.fused_score, 2),
            "fused_score_raw": round(self.fused_score_raw, 4),
            "individual_scores": self.individual_scores,
            "is_match": self.is_match,
            "threshold_percent": round(self.threshold_percent, 2),
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
        }


class BiometricMatchEngine:
    """
    Mathematical similarity matching engine for multi-modal biometric vectors.
    Enforces strict modality boundaries and calculates cosine, Euclidean, and weighted fusion scores.
    """

    # Default similarity thresholds per modality (cosine similarity scale 0.0 to 1.0)
    DEFAULT_THRESHOLDS = {
        BiometricModality.FACE: 0.50,
        BiometricModality.FINGERPRINT: 0.60,
        BiometricModality.BODY_MARK: 0.55,
        BiometricModality.IRIS: 0.70,
    }

    # Configurable weight constants (w_face = 0.50, w_iris = 0.30, w_fingerprint = 0.20 sum to 1.0)
    DEFAULT_WEIGHTS = {
        BiometricModality.FACE: 0.50,
        BiometricModality.IRIS: 0.30,
        BiometricModality.FINGERPRINT: 0.20,
        BiometricModality.BODY_MARK: 0.20,
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
    def fuse_multi_biometrics(
        cls,
        scores_by_modality: Dict[Union[BiometricModality, str], float],
        custom_weights: Optional[Dict[Union[BiometricModality, str], float]] = None,
        threshold_percent: float = 40.0,
    ) -> FusedMatchResult:
        """
        Compute weighted multi-modal biometric confidence score across available modalities.
        Weights are automatically normalized across present biometrics (e.g. 100% on single modality).

        Args:
            scores_by_modality: Mapping of modality (enum or string like "FACE", "IRIS", "FINGERPRINT")
                                to individual similarity score (0.0 to 1.0).
            custom_weights: Optional custom weights mapping per modality.
            threshold_percent: Minimum unified fused score threshold percentage for a match (default: 40.0%).

        Returns:
            FusedMatchResult object containing fused_score (0-100%), individual_scores, is_match, and weights_used.
        """
        if not scores_by_modality:
            return FusedMatchResult(
                fused_score=0.0,
                fused_score_raw=0.0,
                individual_scores={},
                is_match=False,
                threshold_percent=threshold_percent,
                weights_used={},
            )

        # Parse and normalize keys
        normalized_scores: Dict[BiometricModality, float] = {}
        for mod_key, score in scores_by_modality.items():
            if isinstance(mod_key, BiometricModality):
                mod_enum = mod_key
            else:
                try:
                    mod_enum = BiometricModality.from_str(str(mod_key))
                except ValueError:
                    continue
            normalized_scores[mod_enum] = float(np.clip(score, 0.0, 1.0))

        if not normalized_scores:
            return FusedMatchResult(
                fused_score=0.0,
                fused_score_raw=0.0,
                individual_scores={},
                is_match=False,
                threshold_percent=threshold_percent,
                weights_used={},
            )

        # Determine base weight per present modality
        weights_for_present: Dict[BiometricModality, float] = {}

        for mod_enum in normalized_scores.keys():
            w = 0.25
            if custom_weights:
                if mod_enum in custom_weights:
                    w = float(custom_weights[mod_enum])
                elif mod_enum.value in custom_weights:
                    w = float(custom_weights[mod_enum.value])
                elif mod_enum.name in custom_weights:
                    w = float(custom_weights[mod_enum.name])
            else:
                w = cls.DEFAULT_WEIGHTS.get(mod_enum, 0.25)
            weights_for_present[mod_enum] = max(0.0, w)

        total_weight = sum(weights_for_present.values())
        if total_weight == 0:
            total_weight = float(len(normalized_scores))
            normalized_weights = {m: 1.0 / total_weight for m in normalized_scores.keys()}
        else:
            normalized_weights = {m: w / total_weight for m, w in weights_for_present.items()}

        weighted_sum = sum(
            normalized_weights[m] * normalized_scores[m] for m in normalized_scores.keys()
        )
        fused_raw = float(np.clip(weighted_sum, 0.0, 1.0))
        fused_percent = float(round(fused_raw * 100.0, 2))
        is_match = bool(fused_percent >= threshold_percent)

        indiv_scores_dict = {m.value: round(normalized_scores[m], 4) for m in normalized_scores.keys()}
        weights_used_dict = {m.value: round(normalized_weights[m], 4) for m in normalized_scores.keys()}

        return FusedMatchResult(
            fused_score=fused_percent,
            fused_score_raw=round(fused_raw, 4),
            individual_scores=indiv_scores_dict,
            is_match=is_match,
            threshold_percent=threshold_percent,
            weights_used=weights_used_dict,
        )

    @classmethod
    def fuse_scores(
        cls,
        scores_by_modality: Dict[BiometricModality, float],
        custom_weights: Optional[Dict[BiometricModality, float]] = None,
    ) -> float:
        """
        Legacy helper returning raw fused score (0.0 to 1.0).
        Delegates to fuse_multi_biometrics.
        """
        res = cls.fuse_multi_biometrics(scores_by_modality, custom_weights=custom_weights)
        return res.fused_score_raw
