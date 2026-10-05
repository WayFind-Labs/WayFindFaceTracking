---
name: facetracking-context
description: WayFind FaceTracking Project Context - Current state, completed features, architecture, and roadmap.
---

# WayFind FaceTracking Subsystem - Project Context

## Executive Summary
The **WayFind FaceTracking** project is a high-performance facial recognition and biometric vector analysis subsystem within the larger **WayFind Intelligent Platform**. It provides core face detection, 512-dimensional ArcFace feature vector extraction, vector persistence, and fast vector comparison capabilities.

---

## 1. Overall System Architecture & Scope

```
+-----------------------------------------------------------------------+
|                         WayFind Platform                              |
|                                                                       |
|  +-------------------------+         +-----------------------------+  |
|  |   WayFind Spatial &     |         |   WayFind FaceTracking      |  |
|  |   Navigation Core       | <=====> |   Subsystem                 |  |
|  +-------------------------+         +-----------------------------+  |
|                                                     |                 |
|                                       +-------------+-------------+   |
|                                       |                           |   |
|                                       v                           v   |
|                              [Image to Vector]           [Compare Vectors] |
|                              (512-D ArcFace)             (Cosine / L2) |
+-----------------------------------------------------------------------+
```

### Core Responsibilities
- **Biometric Feature Extraction**: Transform raw face images (BGR/RGB/Files) into normalized 512-D floating-point feature embeddings using ArcFace neural network models (`insightface` ONNX backend).
- **Vector Persistence**: Serialize face embeddings along with rich identity metadata into JSON or NumPy (`.npy`) format.
- **Identity Matching**: Calculate high-accuracy similarity metrics between vectors to perform identity verification (1:1 matching) and identification (1:N matching).

---

## 2. Completed Work & Current Implementation

### A. Environment & Setup
- **Virtual Environment**: Python 3.12 isolated environment configured in `venv/`.
- **Environment Verification**: [`test_env.py`](file:///c:/Users/Lenovo/Desktop/FaceTracking/test_env.py) verifies runtime execution.
- **Dependencies ([`requirements.txt`](file:///c:/Users/Lenovo/Desktop/FaceTracking/requirements.txt))**: `insightface`, `onnxruntime`, `opencv-python`, `numpy`, `pillow`, `scipy`.

### B. Core ArcFace Engine ([`arcface_service.py`](file:///c:/Users/Lenovo/Desktop/FaceTracking/arcface_service.py))
- **`ArcFaceService` Class**:
  - `image_to_vector(image_input, save_path=None, metadata=None)`: Detects face, crops/aligns, generates 512-D normalized vector, and optionally saves to disk.
  - `save_vector(vector, save_path, metadata)`: Serializes vector into `.json` (with metadata) or `.npy` format.
  - `load_vector(vector_input)`: Deserializes vector from `.json`, `.npy`, list, or NumPy array.
  - `compare_vectors(vector1, vector2, threshold=0.5)`: Computes:
    - **Cosine Similarity**: $\mathbf{v_1} \cdot \mathbf{v_2}$
    - **Cosine Distance**: $1 - \text{CosineSimilarity}$
    - **Euclidean Distance**: $\|\mathbf{v_1} - \mathbf{v_2}\|_2$
    - **Match Decision & Confidence Score (%)**

### C. CLI & Demonstration Suite
- **[`main.py`](file:///c:/Users/Lenovo/Desktop/FaceTracking/main.py)**: CLI with `convert` and `compare` commands.
- **[`demo.py`](file:///c:/Users/Lenovo/Desktop/FaceTracking/demo.py)**: Automated verification script testing vector extraction, persistence, positive match, and negative match scenarios.

### D. Version Control & Hygiene
- **[`.gitignore`](file:///c:/Users/Lenovo/Desktop/FaceTracking/.gitignore)**: Configured to ignore `venv/`, `__pycache__/`, `.insightface/`, `output/`, and temporary build artifacts.
- **Git Repository**: Initialized, cleaned of untracked index cache (`git rm -r --cached .`), committed, and pushed to GitHub: `https://github.com/anserking/WayFindFaceTracking.git` (`main` branch).

### E. Workflow & Context Guardrails
- **[`workflow-guardrails`](file:///c:/Users/Lenovo/Desktop/FaceTracking/workflow-guardrails/SKILL.md)**: Defines mandatory pre-task `git pull` enforcement, post-phase verification, conventional commit standards, and pre-push context synchronization audits.
- **[`facetracking-context`](file:///c:/Users/Lenovo/Desktop/FaceTracking/context/SKILL.md)**: Contains full project architecture, implementation status, roadmap, and mandatory update guardrails to keep project context synced after every task and before every `git push`.

---

## 3. Pending Roadmap & Next Steps (What Needs To Be Done)

1. **Model Cache Pre-loader (`scripts/download_models.py`)**:
   - Implement an explicit script to pre-download and verify InsightFace model bundles (`buffalo_sc` / `buffalo_l`) to ensure offline operation without downloading on first runtime execution.

2. **Real-Time Video Stream Tracking (`tracking_stream.py`)**:
   - Integrate OpenCV webcam/RTSP stream reader with real-time face bounding box overlays, landmark alignment, and live face embedding extraction.

3. **Fast 1:N Vector Indexing & Search (FAISS / ChromaDB)**:
   - Add FAISS index integration (`faiss-cpu`) to enable sub-millisecond similarity search across large galleries of face vectors.

4. **FastAPI REST Service (`api/app.py`)**:
   - Wrap `ArcFaceService` into a REST API exposing `/convert`, `/compare`, and `/search` endpoints for seamless integration with WayFind web & mobile clients.

5. **GPU Acceleration & Optimization**:
   - Add support for `CUDAExecutionProvider` and `TensorRTExecutionProvider` for high-FPS multi-camera tracking.

---

## 4. Context Skill Update & Maintenance Guardrails

> [!IMPORTANT]
> **MANDATORY Context Synchronization Guardrail**
> 1. **Trigger for Update**: Whenever updating any component, introducing new features/files, refactoring architecture, adjusting dependencies, or completing any user task (especially tasks not previously documented in this skill), the agent MUST update this Context Skill upon completing the work phase.
> 2. **Post-Phase Synchronization**: After completing any implementation phase or task, verify and update:
>    - **Completed Work & Current Implementation (Section 2)**: Record new capabilities, added files, and completed roadmap items.
>    - **Architecture & Dependencies (Section 1 & 2A)**: Update system topology or package dependencies if changed.
>    - **Pending Roadmap (Section 3)**: Update remaining, modified, or newly identified project tasks.
> 3. **Pre-Push Validation Requirement**: Before executing `git push` to upload code to GitHub, check whether the Context Skill accurately reflects the latest codebase state. If updates are needed, update both [`context/SKILL.md`](file:///c:/Users/Lenovo/Desktop/FaceTracking/context/SKILL.md) and [`.agents/skills/context/SKILL.md`](file:///c:/Users/Lenovo/Desktop/FaceTracking/.agents/skills/context/SKILL.md), stage them (`git add .`), and commit before pushing.

