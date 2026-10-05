"""
API Integration Tests for WayFind Biometric Vision Microservice (Port 8003).
Uses starlette/fastapi TestClient with httpx.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from biometrics.store import get_embedding_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_store():
    """Reset the in-memory embedding store before every test."""
    store = get_embedding_store()
    store.clear()
    yield
    store.clear()


def test_health_endpoint():
    """Verify /health returns operational status and modality list."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "WayFindFaceTracking"
    modalities = [m["modality"] for m in data["supported_modalities"]]
    assert "FACE" in modalities
    assert "FINGERPRINT" in modalities
    assert "BODY_MARK" in modalities


def test_root_endpoint():
    """Verify root / returns service landing information."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_direct_vector_embed_and_match():
    """Verify embedding direct vectors and matching across candidates."""
    # 1. Embed query vector
    vec_query = [1.0] + [0.0] * 511
    res1 = client.post(
        "/api/biometrics/embed/direct",
        json={
            "vector": vec_query,
            "modality": "FACE",
            "model_version": "test-v1",
            "person_id": "child_case_001",
            "metadata": {"name": "Test Child A"},
        },
    )
    assert res1.status_code == 201
    query_id = res1.json()["embedding_id"]
    assert res1.json()["dimension"] == 512
    assert res1.json()["person_id"] == "child_case_001"

    # 2. Embed identical candidate vector (Score ~ 1.0)
    vec_cand_match = [1.0] + [0.0] * 511
    res2 = client.post(
        "/api/biometrics/embed/direct",
        json={
            "vector": vec_cand_match,
            "modality": "FACE",
            "model_version": "test-v1",
            "person_id": "found_case_999",
        },
    )
    assert res2.status_code == 201
    cand_match_id = res2.json()["embedding_id"]

    # 3. Embed orthogonal non-match vector (Score ~ 0.0)
    vec_cand_diff = [0.0, 1.0] + [0.0] * 510
    res3 = client.post(
        "/api/biometrics/embed/direct",
        json={
            "vector": vec_cand_diff,
            "modality": "FACE",
            "model_version": "test-v1",
            "person_id": "found_case_888",
        },
    )
    assert res3.status_code == 201
    cand_diff_id = res3.json()["embedding_id"]

    # 4. Perform match
    match_res = client.post(
        "/api/biometrics/match",
        json={
            "query_embedding_id": query_id,
            "candidate_embedding_ids": [cand_match_id, cand_diff_id],
            "modality": "FACE",
            "threshold": 0.50,
            "top_k": 5,
        },
    )
    assert match_res.status_code == 200
    match_data = match_res.json()
    assert match_data["candidates_searched"] == 2
    assert len(match_data["matches"]) == 2

    # Top match must be cand_match_id with score 1.0 and is_match=True
    top_hit = match_data["matches"][0]
    assert top_hit["candidate_embedding_id"] == cand_match_id
    assert top_hit["is_match"] is True
    assert np.isclose(top_hit["similarity_score"], 1.0)

    # Second hit must have score 0.0 and is_match=False
    second_hit = match_data["matches"][1]
    assert second_hit["candidate_embedding_id"] == cand_diff_id
    assert second_hit["is_match"] is False
    assert np.isclose(second_hit["similarity_score"], 0.0)


def test_modality_mismatch_rejection():
    """Verify comparing across different modalities returns 400."""
    # 1. Embed face vector
    res_face = client.post(
        "/api/biometrics/embed/direct",
        json={"vector": [1.0] * 512, "modality": "FACE"},
    )
    face_id = res_face.json()["embedding_id"]

    # 2. Match with modality=FINGERPRINT on FACE query -> 400
    res_mismatch = client.post(
        "/api/biometrics/match",
        json={
            "query_embedding_id": face_id,
            "candidate_embedding_ids": [],
            "modality": "FINGERPRINT",
        },
    )
    assert res_mismatch.status_code == 400
    assert "MODALITY_MISMATCH" in res_mismatch.json()["detail"]


def test_mcp_client_images_compare_shim():
    """Verify /api/images/compare endpoint adheres to WayFindMCPClient contract."""
    # 1. Store candidate vectors in memory
    res_query = client.post(
        "/api/biometrics/embed/direct",
        json={"vector": [1.0] + [0.0] * 511, "modality": "FACE"},
    )
    query_id = res_query.json()["embedding_id"]

    res_cand = client.post(
        "/api/biometrics/embed/direct",
        json={"vector": [1.0] + [0.0] * 511, "modality": "FACE"},
    )
    cand_id = res_cand.json()["embedding_id"]

    # 2. Call MCP shim endpoint
    response = client.post(
        "/api/images/compare",
        json={
            "media_id": query_id,
            "candidate_case_ids": [cand_id, "simulated_case_002"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query_media_id"] == query_id
    assert data["candidates_searched"] == 2
    assert len(data["matches"]) == 2

    # Check first match (real stored vector)
    assert data["matches"][0]["candidate_case_id"] == cand_id
    assert np.isclose(data["matches"][0]["similarity_score"], 1.0)
    assert "facial structure" in data["matches"][0]["match_regions"]

    # Check second match (graceful fallback)
    assert data["matches"][1]["candidate_case_id"] == "simulated_case_002"
    assert data["matches"][1]["similarity_score"] == 0.45


def test_stub_modality_returns_501():
    """Verify uploading to stub modality returns 501 Not Implemented."""
    # Fake multipart file
    files = {"image": ("test.jpg", b"\xff\xd8\xff\xe0fakejpegdata", "image/jpeg")}
    data = {"modality": "FINGERPRINT"}

    response = client.post("/api/biometrics/embed", files=files, data=data)
    assert response.status_code == 501
    assert "not yet operational" in response.json()["detail"]
