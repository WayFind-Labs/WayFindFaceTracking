# WayFindFaceTracking — Multi-Modal Biometric Architecture

## The Core Problem

Right now we have **one modality** (face → 512-D ArcFace vector).
In the future we may have:
- 🔵 **Face** → ArcFace 512-D embedding (current)
- 🟡 **Fingerprint** → Minutiae-based or deep fingerprint embedding (future)
- 🟠 **Body marks** → Tattoo/scar CNN embedding (future)
- 🟣 **Iris** → IrisCode or deep embedding (far future)

The challenge: each modality uses a **different model**, produces **different-sized vectors**,
and requires **different preprocessing pipelines** — but the *downstream logic* (store embedding,
compare embeddings, return similarity) is **identical for all of them**.

---

## Design Patterns

### 1. Strategy Pattern — `BiometricExtractor` Abstract Base

Each modality is a concrete **Strategy** implementing one interface:

```
BiometricExtractor (ABC)
│
├── extract(image: np.ndarray) -> EmbeddingVector
└── modality: BiometricModality

Concrete strategies:
├── ArcFaceExtractor          ← FACE (exists today)
├── FingerprintExtractor      ← FINGERPRINT (future)
└── BodyMarkExtractor         ← BODY_MARK (future)
```

Adding a new modality = write one class, register it. **Zero changes to API, routes, or match engine.**

---

### 2. Registry Pattern — `ExtractorRegistry`

A central registry maps `BiometricModality → Extractor`:

```python
class ExtractorRegistry:
    _extractors: dict[BiometricModality, BiometricExtractor] = {}

    def register(self, modality, extractor): ...
    def get(self, modality) -> BiometricExtractor: ...
    def supported_modalities(self) -> list[BiometricModality]: ...
```

At startup:
```python
registry.register(BiometricModality.FACE, ArcFaceExtractor())
# registry.register(BiometricModality.FINGERPRINT, FingerprintExtractor())  ← uncomment when ready
```

---

### 3. Unified Embedding Model

All modalities produce the same envelope regardless of internal vector size:

```python
@dataclass
class EmbeddingVector:
    modality: BiometricModality   # FACE | FINGERPRINT | BODY_MARK
    vector: np.ndarray            # 512-D for face, 256-D for fingerprint, etc.
    dimension: int
    model_version: str            # "arcface-buffalo_sc-v1", "fingerprint-v1"
    confidence: float             # extraction confidence (0.0 – 1.0)
    embedding_id: str             # UUID — used as the storage/retrieval key
    created_at: datetime
```

**Why this matters:** The match engine, storage layer, and API response schema
never need to know which modality produced the vector. They just store and compare.

---

### 4. Modality-Tagged In-Memory Store (Phase A: API launch)

```
EmbeddingStore
└── _store: dict[str, EmbeddingVector]  ← keyed by embedding_id
└── _by_person: dict[str, list[str]]    ← person_id → [embedding_id, ...]

store.put(embedding_id, EmbeddingVector)
store.get(embedding_id) -> EmbeddingVector
store.get_by_person(person_id) -> list[EmbeddingVector]
```

Matching NEVER crosses modalities:
```
compare(face_embedding_id, fingerprint_embedding_id) → 400 MODALITY_MISMATCH
```

---

## Revised File Structure

```
WayFindFaceTracking/
│
├── api/
│   ├── main.py                       ← FastAPI app (port 8003)
│   ├── dependencies.py               ← DI: get_registry(), get_store()
│   ├── routers/
│   │   ├── health.py                 ← GET /health (lists supported modalities)
│   │   ├── embed.py                  ← POST /api/biometrics/embed
│   │   ├── match.py                  ← POST /api/biometrics/match
│   │   └── images.py                 ← POST /api/images/compare (MCP contract shim)
│   └── schemas/
│       ├── embed.py                  ← EmbedRequest, EmbedResponse
│       └── match.py                  ← MatchRequest, MatchResponse
│
├── biometrics/                       ← Core domain (no FastAPI dependency)
│   ├── __init__.py
│   ├── modality.py                   ← BiometricModality enum
│   ├── base.py                       ← BiometricExtractor ABC + EmbeddingVector
│   ├── registry.py                   ← ExtractorRegistry (singleton)
│   ├── store.py                      ← EmbeddingStore (in-memory, swappable)
│   ├── match_engine.py               ← cosine similarity + ranking (modality-safe)
│   └── extractors/
│       ├── __init__.py
│       ├── face.py                   ← ArcFaceExtractor (wraps arcface_service.py)
│       ├── fingerprint.py            ← FingerprintExtractor (stub, future)
│       └── body_mark.py              ← BodyMarkExtractor (stub, future)
│
├── arcface_service.py                ← (existing, untouched)
├── requirements.txt                  ← + fastapi, uvicorn, python-multipart
└── .env.example
```

---

## API Design (Modality-First)

### POST `/api/biometrics/embed`
```json
Request (multipart/form-data):
  image: <file>
  modality: "FACE" | "FINGERPRINT" | "BODY_MARK"
  person_id: "uuid-optional"   ← for pre-registration (parent registering child)
  metadata: {"name": "Ali", "age": 7}  ← optional JSON string

Response:
{
  "embedding_id": "uuid-v4",
  "modality": "FACE",
  "dimension": 512,
  "model_version": "arcface-buffalo_sc-v1",
  "confidence": 0.97,
  "person_id": "uuid-or-null"
}
```

### POST `/api/biometrics/match`
```json
Request:
{
  "query_embedding_id": "uuid",           ← the found-child's embedding
  "candidate_embedding_ids": ["uuid1", "uuid2"],  ← registered children's embeddings
  "modality": "FACE",                     ← must match for all IDs
  "threshold": 0.5
}

Response:
{
  "matches": [
    {
      "candidate_embedding_id": "uuid1",
      "person_id": "uuid",
      "similarity_score": 0.91,
      "is_match": true,
      "confidence_percent": 91.0
    }
  ],
  "query_embedding_id": "uuid",
  "modality": "FACE",
  "candidates_searched": 2
}
```

### POST `/api/images/compare` (MCP contract shim — unchanged)
This is the existing route `ImageBackendClient` calls. It uses `media_id` 
(which maps to `embedding_id`) internally. **No changes needed to MCP client.**

---

## Two Embed Use Cases You Described

| Scenario | Who calls it | When | Modality |
|---|---|---|---|
| Parent pre-registers child | `ShelterAppUiBackend` → Biometric API | During child registration flow | `FACE` (+ future FINGERPRINT) |
| Found-child intake | `WayFindMCPClient` via `image_match_candidates` | When citizen reports found child | `FACE` |
| Future: body mark intake | MCP tool or backend | During intake if face unclear | `BODY_MARK` |

---

## Future: Multi-Modal Fusion Matching (Phase 4+)

When multiple modalities are available, we can do **fusion scoring**:

```
final_score = (0.6 × face_similarity) + (0.3 × fingerprint_similarity) + (0.1 × body_mark_similarity)
```

Weights are configurable. Missing modalities are excluded from the weighted average.
This requires no API change — just a new `POST /api/biometrics/fuse-match` route.

---

## What We Build Now vs Later

| Component | Now | Later |
|---|---|---|
| `BiometricModality` enum | FACE + stubs for others | Add real extractors |
| `ArcFaceExtractor` | ✅ Full implementation | GPU providers |
| `FingerprintExtractor` | ⬜ Raise `NotImplementedError` | Real model |
| `BodyMarkExtractor` | ⬜ Raise `NotImplementedError` | CNN model |
| `EmbeddingStore` | In-memory dict | Redis / Qdrant |
| `POST /api/biometrics/embed` | ✅ Working (FACE only) | Auto-routes on modality |
| `POST /api/biometrics/match` | ✅ Working (FACE only) | Multi-modal fusion |
| `POST /api/images/compare` | ✅ MCP shim works | Unchanged |
