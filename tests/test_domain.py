"""
Unit verification script for Phase 1: Biometrics Core Domain Layer
Tests:
- BiometricModality enum and parsing
- EmbeddingVector creation and normalization
- ArcFaceExtractor, FingerprintExtractor stub, BodyMarkExtractor stub
- ExtractorRegistry registration and lookup
- EmbeddingStore CRUD, person indexing, modality counts
- BiometricMatchEngine cosine similarity, thresholds, candidate ranking
- Strict MODALITY_MISMATCH prevention
- Multi-modal fusion scoring
"""

import sys
import numpy as np

def run_tests():
    print("=== Testing Biometrics Core Domain Layer ===")

    # 1. Modality Enum
    from biometrics.modality import BiometricModality
    assert BiometricModality.from_str("face") == BiometricModality.FACE
    assert BiometricModality.from_str("FINGERPRINT") == BiometricModality.FINGERPRINT
    assert BiometricModality.from_str("body_mark") == BiometricModality.BODY_MARK
    print(" [PASS] BiometricModality enum & parsing")

    # 2. EmbeddingVector Factory & Normalization
    from biometrics.base import EmbeddingVector
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
    print(" [PASS] EmbeddingVector creation & normalization")

    # 3. Registry & Extractors
    from biometrics.registry import get_extractor_registry
    registry = get_extractor_registry()
    face_ext = registry.get(BiometricModality.FACE)
    fp_ext = registry.get(BiometricModality.FINGERPRINT)
    bm_ext = registry.get(BiometricModality.BODY_MARK)

    assert face_ext.modality == BiometricModality.FACE
    assert face_ext.dimension == 512
    assert face_ext.is_available is True

    assert fp_ext.is_available is False
    assert bm_ext.is_available is False

    # Check stub raises NotImplementedError
    try:
        fp_ext.extract(b"dummy")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass
    print(" [PASS] ExtractorRegistry and Extractor strategies (including stubs)")

    # 4. EmbeddingStore CRUD and Indexing
    from biometrics.store import EmbeddingStore
    store = EmbeddingStore()
    
    # Store face embedding
    store.save(emb)
    assert store.count() == 1
    assert store.count(BiometricModality.FACE) == 1
    assert store.count(BiometricModality.FINGERPRINT) == 0

    # Retrieve by ID
    retrieved = store.get(emb.embedding_id)
    assert retrieved is not None
    assert retrieved.embedding_id == emb.embedding_id

    # Retrieve by Person
    by_person = store.get_by_person("child_123")
    assert len(by_person) == 1
    assert by_person[0].embedding_id == emb.embedding_id

    # Store another vector for same person
    emb2 = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=[1.0] * 512,
        model_version="test-v1",
        person_id="child_123",
    )
    store.save(emb2)
    assert len(store.get_by_person("child_123")) == 2

    # Delete
    store.delete(emb2.embedding_id)
    assert len(store.get_by_person("child_123")) == 1
    print(" [PASS] EmbeddingStore thread-safe operations & secondary indices")

    # 5. BiometricMatchEngine: Similarity & Matching
    from biometrics.match_engine import BiometricMatchEngine

    # Identical vector -> similarity ~ 1.0
    res_exact = BiometricMatchEngine.compare_two(emb, emb)
    assert res_exact.is_match is True
    assert np.isclose(res_exact.similarity_score, 1.0)
    assert np.isclose(res_exact.cosine_distance, 0.0)

    # Orthogonal vector -> similarity ~ 0.0
    ortho_vec = [0.0, 0.0, 1.0] + [0.0] * 509
    emb_ortho = EmbeddingVector.create(
        modality=BiometricModality.FACE,
        vector=ortho_vec,
        model_version="test-v1",
    )
    res_ortho = BiometricMatchEngine.compare_two(emb, emb_ortho)
    assert res_ortho.is_match is False
    assert np.isclose(res_ortho.similarity_score, 0.0)
    assert np.isclose(res_ortho.cosine_distance, 1.0)
    print(" [PASS] BiometricMatchEngine similarity, distances & thresholding")

    # 6. Strict Modality Mismatch Guardrail
    emb_fingerprint = EmbeddingVector.create(
        modality=BiometricModality.FINGERPRINT,
        vector=[1.0] * 256,
        model_version="fp-v1",
    )
    try:
        BiometricMatchEngine.compare_two(emb, emb_fingerprint)
        assert False, "Should have raised ValueError on modality mismatch"
    except ValueError as e:
        assert "MODALITY_MISMATCH" in str(e)
    print(" [PASS] Strict MODALITY_MISMATCH guardrail enforcement")

    # 7. Candidate Ranking
    candidates = [emb_ortho, emb]  # emb is exact match (score 1.0), ortho is score 0.0
    ranked = BiometricMatchEngine.match_candidates(emb, candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].candidate_embedding_id == emb.embedding_id
    assert ranked[0].similarity_score > ranked[1].similarity_score
    print(" [PASS] Multi-candidate ranking (descending similarity)")

    # 8. Multi-Modal Fusion Scoring
    fusion = BiometricMatchEngine.fuse_scores({
        BiometricModality.FACE: 0.90,
        BiometricModality.FINGERPRINT: 0.80,
    })
    # Default weights: FACE 0.5, FP 0.3 -> total weight 0.8
    # (0.5*0.9 + 0.3*0.8) / 0.8 = (0.45 + 0.24) / 0.8 = 0.69 / 0.8 = 0.8625
    assert np.isclose(fusion, 0.8625)
    print(" [PASS] Multi-modal weighted fusion scoring")

    print("\nALL PHASE 1 DOMAIN TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
