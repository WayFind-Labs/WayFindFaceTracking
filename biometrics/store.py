from functools import lru_cache
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from biometrics.base import EmbeddingVector
from biometrics.modality import BiometricModality

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Persistent high-speed vector database store powered by Qdrant.
    Configured for 512-dimensional ArcFace embeddings with Cosine distance.
    Supports in-memory testing (`:memory:`) by default, and cloud/cluster deployment
    via environment variables (`QDRANT_URL`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`).
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        dimension: int = 512,
        client: Optional[QdrantClient] = None,
    ):
        self._lock = threading.RLock()
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "wayfind_biometrics")
        self.dimension = dimension

        if client is not None:
            self._client = client
        else:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_host = os.getenv("QDRANT_HOST")
            qdrant_port = os.getenv("QDRANT_PORT")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if qdrant_url:
                logger.info("Initializing Qdrant client at URL: %s", qdrant_url)
                self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            elif qdrant_host:
                port = int(qdrant_port) if qdrant_port else 6333
                logger.info("Initializing Qdrant client at host: %s:%d", qdrant_host, port)
                self._client = QdrantClient(host=qdrant_host, port=port, api_key=qdrant_api_key)
            else:
                logger.info("Initializing in-memory Qdrant client (:memory:)")
                self._client = QdrantClient(":memory:")

        self._init_collection()

    def _init_collection(self) -> None:
        """Initialize Qdrant collection if it does not already exist."""
        with self._lock:
            if not self._client.collection_exists(self.collection_name):
                logger.info(
                    "Creating Qdrant collection '%s' (dim: %d, distance: COSINE)",
                    self.collection_name,
                    self.dimension,
                )
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )

    @staticmethod
    def _to_qdrant_id(embedding_id: str) -> str:
        """Ensure string embedding ID is formatted as a valid UUID string for Qdrant."""
        try:
            return str(uuid.UUID(embedding_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, embedding_id))

    def _payload_to_embedding(self, payload: Dict[str, Any], vector: Optional[List[float]]) -> EmbeddingVector:
        """Convert a Qdrant point payload and vector back into an EmbeddingVector."""
        mod_str = payload.get("modality", "FACE")
        modality = BiometricModality.from_str(mod_str)
        
        if vector is not None:
            vec_np = np.array(vector, dtype=np.float32)
        else:
            vec_np = np.zeros(payload.get("dimension", self.dimension), dtype=np.float32)

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return EmbeddingVector(
            embedding_id=payload.get("embedding_id", str(uuid.uuid4())),
            modality=modality,
            vector=vec_np,
            dimension=payload.get("dimension", len(vec_np)),
            model_version=payload.get("model_version", "unknown"),
            confidence=float(payload.get("confidence", 1.0)),
            person_id=payload.get("person_id"),
            metadata=metadata,
            created_at=payload.get("created_at", payload.get("timestamp")),
        )

    def save(self, embedding: EmbeddingVector) -> str:
        """
        Store an embedding vector and payload metadata into Qdrant collection.
        Includes child_name, status_type, contact_info, location, and timestamp.
        """
        with self._lock:
            eid = embedding.embedding_id
            qid = self._to_qdrant_id(eid)
            vec_list = embedding.vector.tolist()

            meta = embedding.metadata or {}
            payload = {
                "embedding_id": eid,
                "person_id": embedding.person_id,
                "modality": embedding.modality.value,
                "dimension": embedding.dimension,
                "model_version": embedding.model_version,
                "confidence": float(embedding.confidence),
                "created_at": embedding.created_at,
                "child_name": meta.get("child_name") or meta.get("name"),
                "status_type": meta.get("status_type", "missing"),
                "contact_info": meta.get("contact_info"),
                "location": meta.get("location"),
                "timestamp": meta.get("timestamp") or embedding.created_at,
                "metadata": meta,
            }

            point = PointStruct(id=qid, vector=vec_list, payload=payload)
            self._client.upsert(collection_name=self.collection_name, points=[point])

            logger.debug(
                "Stored embedding %s (qdrant_id: %s) for modality %s into Qdrant",
                eid,
                qid,
                embedding.modality.value,
            )
            return eid

    def get(self, embedding_id: str) -> Optional[EmbeddingVector]:
        """Retrieve an embedding by ID from Qdrant."""
        with self._lock:
            qid = self._to_qdrant_id(embedding_id)
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[qid],
                with_vectors=True,
            )
            if not points:
                return None
            return self._payload_to_embedding(points[0].payload, points[0].vector)

    def get_many(self, embedding_ids: List[str]) -> List[EmbeddingVector]:
        """Retrieve multiple embeddings by IDs in requested order (skips missing)."""
        with self._lock:
            qids = [self._to_qdrant_id(eid) for eid in embedding_ids]
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=qids,
                with_vectors=True,
            )
            point_map = {}
            for p in points:
                if p.payload and "embedding_id" in p.payload:
                    point_map[p.payload["embedding_id"]] = p

            results = []
            for eid in embedding_ids:
                if eid in point_map:
                    p = point_map[eid]
                    results.append(self._payload_to_embedding(p.payload, p.vector))
            return results

    def get_by_person(
        self, person_id: str, modality: Optional[BiometricModality] = None
    ) -> List[EmbeddingVector]:
        """Retrieve all embeddings associated with a person ID, optionally filtered by modality."""
        with self._lock:
            must_conditions = [FieldCondition(key="person_id", match=MatchValue(value=person_id))]
            if modality is not None:
                must_conditions.append(
                    FieldCondition(key="modality", match=MatchValue(value=modality.value))
                )

            scroll_filter = Filter(must=must_conditions)
            scroll_res, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                with_vectors=True,
                limit=1000,
            )

            return [self._payload_to_embedding(p.payload, p.vector) for p in scroll_res]

    def delete(self, embedding_id: str) -> bool:
        """Remove an embedding point from Qdrant by ID."""
        with self._lock:
            existing = self.get(embedding_id)
            if not existing:
                return False

            qid = self._to_qdrant_id(embedding_id)
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=[qid],
            )
            return True

    def count(self, modality: Optional[BiometricModality] = None) -> int:
        """Count stored points in Qdrant, optionally filtered by modality."""
        with self._lock:
            if modality is None:
                return self._client.count(collection_name=self.collection_name).count

            count_filter = Filter(
                must=[FieldCondition(key="modality", match=MatchValue(value=modality.value))]
            )
            return self._client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
            ).count

    def clear(self) -> None:
        """Clear all points from the Qdrant collection."""
        with self._lock:
            if self._client.collection_exists(self.collection_name):
                self._client.delete_collection(self.collection_name)
            self._init_collection()

    def search(
        self,
        query_vector: Union[List[float], np.ndarray, EmbeddingVector],
        modality: Optional[BiometricModality] = None,
        top_k: int = 10,
        threshold: float = 0.40,
    ) -> List[Dict[str, Any]]:
        """
        Perform HNSW Cosine similarity search in Qdrant.
        
        Args:
            query_vector: Vector list, numpy array, or EmbeddingVector instance.
            modality: Optional modality filter (e.g. BiometricModality.FACE).
            top_k: Maximum number of candidate hits to return.
            threshold: Minimum similarity score threshold (default: 0.40).
            
        Returns:
            List of candidate hit dicts containing scores, match decisions, and metadata.
        """
        with self._lock:
            if isinstance(query_vector, EmbeddingVector):
                q_vec = query_vector.vector.tolist()
            elif isinstance(query_vector, np.ndarray):
                q_vec = query_vector.astype(np.float32).flatten().tolist()
            elif isinstance(query_vector, list):
                q_vec = [float(x) for x in query_vector]
            else:
                raise TypeError("query_vector must be a list, numpy array, or EmbeddingVector.")

            must_conditions = []
            if modality is not None:
                mod_str = modality.value if isinstance(modality, BiometricModality) else str(modality)
                must_conditions.append(
                    FieldCondition(key="modality", match=MatchValue(value=mod_str))
                )

            query_filter = Filter(must=must_conditions) if must_conditions else None

            res = self._client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=threshold,
                with_vectors=True,
            )

            hits = []
            for point in res.points:
                score = float(point.score)
                payload = point.payload or {}
                hits.append({
                    "embedding_id": payload.get("embedding_id"),
                    "person_id": payload.get("person_id"),
                    "modality": payload.get("modality"),
                    "similarity_score": round(score, 4),
                    "cosine_distance": round(1.0 - score, 4),
                    "is_match": bool(score >= threshold),
                    "threshold": threshold,
                    "child_name": payload.get("child_name"),
                    "status_type": payload.get("status_type"),
                    "contact_info": payload.get("contact_info"),
                    "location": payload.get("location"),
                    "timestamp": payload.get("timestamp"),
                    "metadata": payload.get("metadata", {}),
                })

            return hits


# Convenient aliases
BiometricStore = EmbeddingStore


@lru_cache(maxsize=1)
def get_embedding_store() -> EmbeddingStore:
    """Singleton getter for global EmbeddingStore."""
    return EmbeddingStore()


def get_biometric_store() -> EmbeddingStore:
    """Alias singleton getter for EmbeddingStore."""
    return get_embedding_store()
