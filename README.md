# WayFind FaceTracking & Biometric Vision Microservice

Standalone **Multi-Modal Biometric Extraction, Storage, and Vector Matching Engine** for the **WayFind Missing Child Platform**.

Runs as a decoupled microservice on **Port 8003**, consuming images and returning normalized 512-D ArcFace embedding vectors, similarity metrics, and candidate match scores.

---

## 🚀 Key Features

- **Multi-Modal Extractor Pipeline**: Implements the **Strategy Pattern** via `BiometricExtractor` and **Registry Pattern** via `ExtractorRegistry` for zero-change addition of new modalities (Face ✅, Fingerprint ⬜, Body Marks ⬜, Iris ⬜).
- **FastAPI REST API (Port 8003)**:
  - `GET /health` — Liveness and registered modality availability.
  - `POST /api/biometrics/embed` — Multipart image upload to normalized embedding vector.
  - `POST /api/biometrics/embed/direct` — Direct pre-computed vector registration.
  - `POST /api/biometrics/match` — Modality-safe candidate vector ranking and similarity computation.
  - `POST /api/images/compare` — MCP Client compatibility shim consumed directly by `WayFindMCPClient`'s `ImageBackendClient`.
- **Modality Safety Guardrails**: Prevents cross-modality vector comparison (`MODALITY_MISMATCH` → HTTP 400).
- **In-Memory Store**: Thread-safe caching with secondary index by person ID and modality.
- **Fusion Engine**: Multi-modal weighted fusion scoring (`BiometricMatchEngine.fuse_scores`).

---

## 📦 Installation & Setup

```bash
# Clone and enter project
git clone https://github.com/anserking/WayFindFaceTracking.git
cd WayFindFaceTracking

# Install dependencies with uv (or pip)
uv pip install -r requirements.txt
```

---

## 🏃 Running the Service

### Start FastAPI Server (Port 8003)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8003 --reload
```
Interactive Swagger API docs available at: `http://localhost:8003/docs`

### Run CLI Tools
```bash
# Convert image to vector
python main.py convert --image sample.jpg --save output/vector.json

# Compare two vectors or images
python main.py compare --vec1 output/vector1.json --vec2 output/vector2.json --threshold 0.5
```

---

## 🧪 Running Tests

```bash
uv run pytest -v tests/
```

---

## 🔗 Integration with WayFind MCP Client

Set the following in `WayFindMCPClient/.env`:
```env
IMAGE_BACKEND_URL=http://localhost:8003
```
Zero code changes needed in `WayFindMCPClient`!
