"""
Unit test suite for Biometrics Core Domain Layer (pytest).
"""

import numpy as np
import pytest

from biometrics.base import EmbeddingVector
from biometrics.match_engine import BiometricMatchEngine
from biometrics.modality import BiometricModality
from biometrics.registry import get_extractor_registry
from biometrics.store import EmbeddingStore


def test_modality_enum_and_parsing():
    assert BiometricModality.from_str("face") == BiometricModality.FACE
    assert BiometricModality.from_str("FINGERPRINT") == BiometricModality.FINGERPRINT
    assert BiometricModality.from_str("body_mark") == BiometricModality.BODY_MARK
    with pytest.raises(ValueError, match="Unsupported biometric modality"):
        BiometricModality.from_str("INVALID_MODALITY")


def test_embedding_vector_creation_and_normalization():
    raw_vec = [3.0, 4.0] + [0.0] * 510  # magnitude = 5.0
    emb = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=raw_vec,
        model_version="test-v1",
        confidence=0.95,
        person_id="child_123",
        metadata={"source": "test"},
    )
    assert emb.dimension == 512
    assert np.isclose(np.linalg.norm(emb.vector), 1.0)
    assert np.isclose(emb.vector[0], 3.0 / 5.0)
    assert emb.person_id == "child_123"


def test_registry_and_extractor_strategies():
    registry = get_extractor_registry()
    face_ext = registry.get(BiometricModality.FACE)
    fp_ext = registry.get(BiometricModality.FINGERPRINT)
    bm_ext = registry.get(BiometricModality.BODY_MARK)

    assert face_ext.modality == BiometricModality.FACE
    assert face_ext.dimension == 512
    assert face_ext.is_available is True

    assert fp_ext.is_available is False
    assert bm_ext.is_available is False

    with pytest.raises(NotImplementedError):
        fp_ext.extract(b"dummy")


def test_embedding_store_operations():
    store = EmbeddingStore()
    emb = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=[1.0] + [0.0] * 511,
        model_version="test-v1",
        person_id="child_123",
    )

    store.save(emb)
    assert store.count() == 1
    assert store.count(BiometricModality.FACE) == 1
    assert store.count(BiometricModality.FINGERPRINT) == 0

    retrieved = store.get(emb.embedding_id)
    assert retrieved is not None
    assert retrieved.embedding_id == emb.embedding_id

    by_person = store.get_by_person("child_123")
    assert len(by_person) == 1
    assert by_person[0].embedding_id == emb.embedding_id

    emb2 = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=[1.0] * 512,
        model_version="test-v1",
        person_id="child_123",
    )
    store.save(emb2)
    assert len(store.get_by_person("child_123")) == 2

    store.delete(emb2.embedding_id)
    assert len(store.get_by_person("child_123")) == 1


def test_qdrant_search_and_metadata():
    store = EmbeddingStore()
    emb = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=[1.0] + [0.0] * 511,
        model_version="test-v1",
        person_id="child_456",
        metadata={
            "child_name": "Johnny Doe",
            "status_type": "missing",
            "contact_info": "555-0199",
            "location": "Shelter A",
            "timestamp": "2026-10-05T00:00:00Z",
        },
    )
    store.save(emb)

    hits = store.search(
        query_vector=[1.0] + [0.0] * 511,
        modality=BiometricModality.FACE,
        top_k=5,
        threshold=0.40,
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit["embedding_id"] == emb.embedding_id
    assert hit["person_id"] == "child_456"
    assert hit["child_name"] == "Johnny Doe"
    assert hit["status_type"] == "missing"
    assert hit["contact_info"] == "555-0199"
    assert hit["location"] == "Shelter A"
    assert hit["is_match"] is True
    assert hit["similarity_score"] >= 0.40



def test_match_engine_similarity_and_mismatch_guardrail():
    emb1 = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=[1.0] + [0.0] * 511,
        model_version="test-v1",
    )
    # Exact match
    res_exact = BiometricMatchEngine.compare_two(emb1, emb1)
    assert res_exact.is_match is True
    assert np.isclose(res_exact.similarity_score, 1.0)
    assert np.isclose(res_exact.cosine_distance, 0.0)

    # Orthogonal
    emb_ortho = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=[0.0, 1.0] + [0.0] * 510,
        model_version="test-v1",
    )
    res_ortho = BiometricMatchEngine.compare_two(emb1, emb_ortho)
    assert res_ortho.is_match is False
    assert np.isclose(res_ortho.similarity_score, 0.0)

    # Modality mismatch
    emb_fp = EmbeddingVector.create(
        modality=BiometricModality.FINGERPRINT,
        vector=[1.0] * 256,
        model_version="fp-v1",
    )
    with pytest.raises(ValueError, match="MODALITY_MISMATCH"):
        BiometricMatchEngine.compare_two(emb1, emb_fp)


def test_fusion_scoring():
    fusion = BiometricMatchEngine.fuse_scores({
        BiometricModality.FACE: 0.90,
        BiometricModality.FINGERPRINT: 0.80,
    })
    assert np.isclose(fusion, 0.8625)
