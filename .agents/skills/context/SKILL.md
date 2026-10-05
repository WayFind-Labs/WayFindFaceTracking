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

## 2. Multi-Modal Biometric Architecture (Design Finalized — Implementation Pending)

### Design Patterns Chosen

#### Strategy Pattern — `BiometricExtractor` ABC
Each biometric modality is a concrete strategy implementing a single interface:
```
BiometricExtractor (ABC)
├── modality: BiometricModality
└── extract(image: np.ndarray) -> EmbeddingVector

Concrete Implementations:
├── ArcFaceExtractor      ← FACE (current, wraps arcface_service.py)
├── FingerprintExtractor  ← FINGERPRINT (future — stub raises NotImplementedError)
└── BodyMarkExtractor     ← BODY_MARK (future — stub raises NotImplementedError)
```

#### Registry Pattern — `ExtractorRegistry`
Central singleton mapping `BiometricModality → BiometricExtractor`.
Adding a new modality = implement extractor + one `registry.register()` call.
Zero changes to API routes, match engine, or MCP client.

### Supported Modalities
| Modality | Status | Model | Vector Dim |
|---|---|---|---|
| `FACE` | ✅ Active | InsightFace ArcFace (`buffalo_sc`) | 512-D |
| `FINGERPRINT` | ⬜ Stub | TBD | TBD |
| `BODY_MARK` | ⬜ Stub | TBD (CNN) | TBD |
| `IRIS` | ⬜ Not started | TBD | TBD |

### Unified Embedding Envelope
All modalities produce an `EmbeddingVector` dataclass containing:
- `modality`, `vector`, `dimension`, `model_version`, `confidence`, `embedding_id`, `created_at`

### Modality Safety Rule
**Embeddings NEVER cross-compare across modalities.**
`compare(FACE_embedding, FINGERPRINT_embedding)` → `400 MODALITY_MISMATCH`

---

## 3. Completed Work & Current Implementation

### A. Core ArcFace Engine (`arcface_service.py`)
- `ArcFaceService` class with `image_to_vector()`, `save_vector()`, `load_vector()`, `compare_vectors()`
- InsightFace `buffalo_sc` model, CPU execution provider, 512-D normalized embeddings
- Cosine similarity, cosine distance, Euclidean distance, match decision + confidence

### B. CLI & Demo Suite
- `main.py`: CLI with `convert` and `compare` commands
- `demo.py`: Automated verification (vector extraction, persistence, positive/negative match)

### C. Environment & Hygiene
- Python 3.12 venv, `requirements.txt`
- `.gitignore` covering `venv/`, `__pycache__/`, `.insightface/`, `output/`, `*.npy`, `*.json`
- Git: `https://github.com/anserking/WayFindFaceTracking.git` (`main` branch)

### D. Workflow & Context Guardrails
- `workflow-guardrails` skill with pre-task pull, post-phase commit/push, conventional commits
- Context skill maintenance guardrails (this file + `context/SKILL.md`)

---

## 4. Planned File Structure (Next Implementation Phase)

```
WayFindFaceTracking/
├── api/                              ← FastAPI HTTP layer (port 8003)
│   ├── main.py
│   ├── dependencies.py
│   ├── routers/
│   │   ├── health.py                 ← GET /health
│   │   ├── embed.py                  ← POST /api/biometrics/embed
│   │   ├── match.py                  ← POST /api/biometrics/match
│   │   └── images.py                 ← POST /api/images/compare (MCP shim)
│   └── schemas/
│       ├── embed.py
│       └── match.py
├── biometrics/                       ← Pure domain layer
│   ├── modality.py                   ← BiometricModality enum
│   ├── base.py                       ← BiometricExtractor ABC + EmbeddingVector
│   ├── registry.py                   ← ExtractorRegistry singleton
│   ├── store.py                      ← In-memory EmbeddingStore
│   ├── match_engine.py               ← Cosine similarity + ranking
│   └── extractors/
│       ├── face.py                   ← ArcFaceExtractor
│       ├── fingerprint.py            ← Stub
│       └── body_mark.py              ← Stub
├── arcface_service.py                ← Existing (untouched)
├── requirements.txt                  ← + fastapi, uvicorn, python-multipart
└── .env.example
```

---

## 5. Pending Roadmap

1. **FastAPI Biometric Microservice (`api/` + `biometrics/`)** — ⬜ IN DESIGN
   - Multi-modal Strategy + Registry pattern architecture
   - Phase A: FACE modality fully working; FINGERPRINT and BODY_MARK as stubs

2. **Model Cache Pre-loader (`scripts/download_models.py`)** — ⬜

3. **Real-Time Video Stream Tracking** — ⬜

4. **Fast 1:N Vector Indexing (FAISS / Qdrant)** — ⬜
   - Replace in-memory store with persistent vector DB

5. **Multi-Modal Fusion Matching** — ⬜ (Phase 4+)
   - `POST /api/biometrics/fuse-match` with weighted fusion scoring

6. **GPU Acceleration** — ⬜

---

## 6. Context Skill Maintenance Guardrails

> [!IMPORTANT]
> **MANDATORY: Update this file and `context/SKILL.md` together before every `git push`.**
> - Section 3 (Completed Work): record new capabilities and files added
> - Section 5 (Roadmap): mark completed/pending items
> - Section 2 (Architecture): update if modalities, patterns, or structure changes
