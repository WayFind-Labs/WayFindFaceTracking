---
name: wayfind-facetracking-context
description: >
  WayFind FaceTracking Subsystem — project context, architecture, implemented
  features, multi-modal biometric design decisions, and pending roadmap.
---

# WayFind FaceTracking — Project Context & Architecture Reference

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

### Supported Modalities (Enum)
| Modality | Status | Model | Vector Dim |
|---|---|---|---|
| `FACE` | ✅ Active | InsightFace ArcFace (`buffalo_sc`) | 512-D |
| `FINGERPRINT` | ⬜ Stub | TBD | TBD |
| `BODY_MARK` | ⬜ Stub | TBD (CNN) | TBD |
| `IRIS` | ⬜ Not started | TBD | TBD |

### Unified Embedding Envelope
All modalities produce:
```python
@dataclass
class EmbeddingVector:
    modality: BiometricModality
    vector: np.ndarray        # 512-D for face, varies by modality
    dimension: int
    model_version: str        # "arcface-buffalo_sc-v1"
    confidence: float         # 0.0 – 1.0 extraction confidence
    embedding_id: str         # UUID — storage/retrieval key
    created_at: datetime
```

### Modality Safety Rule
**Embeddings NEVER cross-compare across modalities.**
`compare(FACE_embedding, FINGERPRINT_embedding)` → `400 MODALITY_MISMATCH`

### Embed Call Use Cases
| Scenario | Caller | When | Modality |
|---|---|---|---|
| Parent pre-registers child | ShelterAppUiBackend | Child registration flow | FACE (+ future FINGERPRINT) |
| Found-child intake | MCP `image_match_candidates` | Citizen reports found child | FACE |
| Body mark intake | MCP tool or backend | When face is unclear/unavailable | BODY_MARK (future) |

---

## 3. Planned File Structure (`api/` + `biometrics/`)

```
WayFindFaceTracking/
│
├── api/                              ← FastAPI HTTP layer (port 8003)
│   ├── main.py                       ← App entrypoint
│   ├── dependencies.py               ← DI: get_registry(), get_store()
│   ├── routers/
│   │   ├── health.py                 ← GET /health
│   │   ├── embed.py                  ← POST /api/biometrics/embed
│   │   ├── match.py                  ← POST /api/biometrics/match
│   │   └── images.py                 ← POST /api/images/compare (MCP shim)
│   └── schemas/
│       ├── embed.py                  ← EmbedRequest, EmbedResponse
│       └── match.py                  ← MatchRequest, MatchResponse
│
├── biometrics/                       ← Pure domain layer (no FastAPI dep)
│   ├── modality.py                   ← BiometricModality enum
│   ├── base.py                       ← BiometricExtractor ABC + EmbeddingVector
│   ├── registry.py                   ← ExtractorRegistry singleton
│   ├── store.py                      ← In-memory EmbeddingStore (swappable)
│   ├── match_engine.py               ← Cosine similarity + ranking (modality-safe)
│   └── extractors/
│       ├── face.py                   ← ArcFaceExtractor (wraps arcface_service.py)
│       ├── fingerprint.py            ← FingerprintExtractor (stub)
│       └── body_mark.py              ← BodyMarkExtractor (stub)
│
├── arcface_service.py                ← Existing ArcFace engine (untouched)
├── main.py                           ← Existing CLI (untouched)
├── demo.py                           ← Existing demo suite (untouched)
├── requirements.txt                  ← + fastapi, uvicorn, python-multipart
└── .env.example
```

---

## 4. Completed Work

### A. Core ArcFace Engine (`arcface_service.py`)
- `ArcFaceService` class with `image_to_vector()`, `save_vector()`, `load_vector()`, `compare_vectors()`
- InsightFace `buffalo_sc` model, CPU execution provider
- CLI (`main.py`) with `convert` and `compare` commands
- Demo suite (`demo.py`) for end-to-end verification

### B. Environment & Hygiene
- Python 3.12 venv, `requirements.txt` with `insightface`, `onnxruntime`, `opencv-python`, `numpy`, `pillow`, `scipy`
- `.gitignore` covering `venv/`, `__pycache__/`, `.insightface/`, `output/`, `*.npy`, `*.json`
- Git repository: `https://github.com/anserking/WayFindFaceTracking.git` (`main` branch)

### C. Workflow & Context Guardrails
- `workflow-guardrails` skill: pre-task pull, post-phase commit/push, conventional commits
- Context skill maintenance guardrails

---

## 5. Pending Roadmap

1. **FastAPI Biometric Microservice (`api/` + `biometrics/`)** — ⬜ IN DESIGN
   - Multi-modal architecture with Strategy + Registry patterns
   - Phase A: FACE modality fully working, stubs for FINGERPRINT and BODY_MARK
   - Phase B: Update context + commit

2. **Model Cache Pre-loader (`scripts/download_models.py`)** — ⬜
   - Pre-download `buffalo_sc` / `buffalo_l` for offline operation

3. **Real-Time Video Stream Tracking (`tracking_stream.py`)** — ⬜
   - OpenCV webcam/RTSP with live face bounding box + embedding extraction

4. **Fast 1:N Vector Indexing (FAISS / Qdrant)** — ⬜
   - Replace in-memory `EmbeddingStore` with persistent vector DB

5. **Multi-Modal Fusion Matching** — ⬜ (Phase 4+)
   - Weighted fusion: `score = 0.6×face + 0.3×fingerprint + 0.1×body_mark`
   - `POST /api/biometrics/fuse-match` route

6. **GPU Acceleration** — ⬜
   - `CUDAExecutionProvider` and `TensorRTExecutionProvider`

---

## 6. Context Skill Maintenance Guardrails

> [!IMPORTANT]
> **MANDATORY: Update this file and `.agents/skills/context/SKILL.md` together before every `git push`.**
> - Section 4 (Completed Work): record new capabilities and files
> - Section 5 (Roadmap): update completed/pending items
> - Section 2 (Architecture): update if modalities, patterns, or structure changes
