---
name: facetracking-context
description: WayFind FaceTracking Project Context - Current state, completed features, multi-modal biometric architecture, and roadmap.
---

# WayFind FaceTracking Subsystem — Project Context

## Executive Summary

The **WayFind FaceTracking** project is the standalone **Biometric Vision Microservice**
within the larger WayFind intelligent missing-child platform. It runs on **port 8003**,
accepts image uploads via HTTP, and returns biometric embedding vectors and similarity
scores. It is decoupled from all other services via `ImageBackendClient` in `WayFindMCPClient`.

---

## 1. Role in the WayFind Platform

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       Next.js PWA Client (ShelterApp ui)                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ REST / WebSockets
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Core Backend (ShelterAppUiBackend)                    │
│   • Temporal Window Engine, Multi-Attribute Scoring Engine              │
│   • Guardian Registry, SOS Beacons, Found Child Cases                  │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼ (Top 3–5 candidates; only if significance_score >= 0.60)
┌─────────────────────────────────────────────────────────────────────────┐
│         WayFindMCPClient (LangGraph Agent + FastMCP Server)             │
│   • image_match_candidates MCP tool                                     │
│   • ImageBackendClient → POST /api/images/compare                       │
│   • IMAGE_BACKEND_URL env var points to this service                    │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │ HTTP (IMAGE_BACKEND_URL=http://localhost:8003)
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│       WayFindFaceTracking — Biometric Vision Microservice (Port 8003)   │
│                                                                         │
│   POST /api/biometrics/embed   → Image → EmbeddingVector (by modality) │
│   POST /api/biometrics/match   → Embedding vs Candidates → Ranked hits  │
│   POST /api/images/compare     → MCP contract shim (media_id based)    │
│   GET  /health                 → Liveness + supported modalities        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Significance Gating (enforced by MCP client, not this service):**
Biometric scanning is ONLY triggered when the deterministic metadata scoring
engine produces `significance_score >= 0.60` on pre-filtered candidates.
This service never performs blind full-database scans.

---

## 2. Multi-Modal Biometric Architecture

### Design Patterns Implemented

#### Strategy Pattern — `BiometricExtractor` ABC (`biometrics/base.py`)
Each biometric modality is a concrete strategy implementing a single interface:
```
BiometricExtractor (ABC)
├── modality: BiometricModality
└── extract(image: Union[str, np.ndarray, bytes]) -> EmbeddingVector

Concrete Implementations:
├── ArcFaceExtractor      ← FACE (wraps arcface_service.py)
├── FingerprintExtractor  ← FINGERPRINT (roadmap stub)
└── BodyMarkExtractor     ← BODY_MARK (roadmap stub)
```

#### Registry Pattern — `ExtractorRegistry` (`biometrics/registry.py`)
Central singleton mapping `BiometricModality → BiometricExtractor`.
Adding a new modality = implement extractor + one `registry.register()` call.
Zero changes to API routes, match engine, or MCP client.

### Supported Modalities
| Modality | Status | Model | Vector Dim |
|---|---|---|---|
| `FACE` | ✅ Operational | InsightFace ArcFace (`buffalo_sc`) | 512-D |
| `FINGERPRINT` | ⬜ Stub (501) | TBD | 256-D |
| `BODY_MARK` | ⬜ Stub (501) | TBD (CNN) | 128-D |
| `IRIS` | ⬜ Roadmap | TBD | TBD |

### Unified Embedding Envelope
All modalities produce `EmbeddingVector` dataclass containing:
- `embedding_id`, `modality`, `vector` (normalized 1D ndarray), `dimension`, `model_version`, `confidence`, `person_id`, `metadata`, `created_at`

### Modality Safety Rule
**Embeddings NEVER cross-compare across modalities.**
`compare(FACE_embedding, FINGERPRINT_embedding)` → `400 MODALITY_MISMATCH`

---

## 3. Implemented File Structure

```
WayFindFaceTracking/
│
├── api/                              ← FastAPI HTTP layer (port 8003)
│   ├── main.py                       ← App entrypoint with CORS & error handlers
│   ├── dependencies.py               ← DI: get_registry(), get_store()
│   ├── routers/
│   │   ├── health.py                 ← GET /health, GET /
│   │   ├── embed.py                  ← POST /api/biometrics/embed, /embed/direct
│   │   ├── match.py                  ← POST /api/biometrics/match (modality-safe)
│   │   └── images.py                 ← POST /api/images/compare (MCP client shim)
│   └── schemas/
│       ├── embed.py                  ← EmbedResponse, DirectVectorEmbedRequest
│       └── match.py                  ← BiometricMatchRequest, ImageCompareRequest
│
├── biometrics/                       ← Pure domain layer (no FastAPI dep)
│   ├── modality.py                   ← BiometricModality enum
│   ├── base.py                       ← BiometricExtractor ABC + EmbeddingVector
│   ├── registry.py                   ← ExtractorRegistry singleton
│   ├── store.py                      ← In-memory EmbeddingStore (thread-safe)
│   ├── match_engine.py               ← Cosine similarity, distance, fusion scoring
│   └── extractors/
│       ├── face.py                   ← ArcFaceExtractor (wraps arcface_service.py)
│       ├── fingerprint.py            ← FingerprintExtractor (stub)
│       └── body_mark.py              ← BodyMarkExtractor (stub)
│
├── tests/
│   ├── test_domain.py                ← Core domain & fusion unit tests
│   └── test_api.py                   ← FastAPI TestClient endpoint integration tests
│
├── arcface_service.py                ← Existing ArcFace engine
├── main.py                           ← Existing CLI
├── demo.py                           ← Existing demo suite
├── requirements.txt                  ← Updated dependencies
└── .env.example                      ← Service configuration template
```

---

## 4. Completed Work

### A. Core ArcFace Engine (`arcface_service.py`)
- `ArcFaceService` class with `image_to_vector()`, `save_vector()`, `load_vector()`, `compare_vectors()`
- InsightFace `buffalo_sc` model, CPU execution provider
- CLI (`main.py`) with `convert` and `compare` commands
- Demo suite (`demo.py`) for end-to-end verification

### B. Multi-Modal Domain Layer (`biometrics/` — Phase 1)
- `BiometricModality` enum with case-insensitive parsing
- `EmbeddingVector` unified envelope with L2 auto-normalization
- `BiometricExtractor` ABC (Strategy Pattern)
- `ArcFaceExtractor`, `FingerprintExtractor`, `BodyMarkExtractor`
- `ExtractorRegistry` singleton with dynamic modality registration
- `EmbeddingStore` thread-safe in-memory cache with secondary person & modality indices
- `BiometricMatchEngine` with strict `MODALITY_MISMATCH` protection and multi-modal weighted fusion scoring

### C. FastAPI Biometric Microservice (`api/` — Phase 2)
- `GET /health`: Operational liveness & modality availability
- `POST /api/biometrics/embed`: Multipart image extraction & storage
- `POST /api/biometrics/embed/direct`: Direct pre-computed vector registration
- `POST /api/biometrics/match`: Modality-safe candidate ranking
- `POST /api/images/compare`: Backward-compatible MCP client shim matching `ImageBackendClient`
- 100% test coverage passing across all 12 domain and API test cases

### D. Persistent Qdrant Vector Database (`biometrics/store.py` — Phase 3)
- `EmbeddingStore` backed by `qdrant-client` targeting `"wayfind_biometrics"` 512-D Cosine collection
- Supports local in-memory testing (`:memory:`) by default with environment overrides (`QDRANT_URL`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`)
- Rich metadata payload support (`child_name`, `status_type`, `contact_info`, `location`, `timestamp`)
- HNSW Cosine similarity search with configurable threshold (default `0.40`)
- 100% test coverage passing across all 13 domain and API test cases

---

## 5. Pending Roadmap

1. **Model Cache Pre-loader (`scripts/download_models.py`)** — ⬜
   - Pre-download `buffalo_sc` / `buffalo_l` for offline operation

2. **Real-Time Video Stream Tracking (`tracking_stream.py`)** — ⬜
   - OpenCV webcam/RTSP with live face bounding box + embedding extraction

3. **Persistent Qdrant Vector DB Integration** — ✅
   - Upgraded `EmbeddingStore` to Qdrant HNSW vector search with metadata payload support

4. **Multi-Modal Fusion Matching Endpoint** — ⬜ (Phase 4+)
   - Dedicated `POST /api/biometrics/fuse-match` route for combined multi-sensor reports


5. **GPU Acceleration** — ⬜
   - `CUDAExecutionProvider` and `TensorRTExecutionProvider`

---

## 6. Context Skill Maintenance Guardrails

> [!IMPORTANT]
> **MANDATORY: Update this file and `context/SKILL.md` together before every `git push`.**
> - Section 4 (Completed Work): record new capabilities and files
> - Section 5 (Roadmap): update completed/pending items
> - Section 2 (Architecture): update if modalities, patterns, or structure changes
