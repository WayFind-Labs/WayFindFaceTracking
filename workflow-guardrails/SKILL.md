---
name: workflow-guardrails
description: Mandatory workflow guardrails for Git synchronization, pre-task pulling, and post-phase pushing for the FaceTracking project.
---

# FaceTracking Workflow & Development Guardrails

This skill defines mandatory operational guardrails and development workflows for the WayFind FaceTracking project.

---

## 1. MANDATORY Pre-Execution Guardrail: Pull Latest Changes

> [!IMPORTANT]
> **RULE 1: Always Pull Before Starting Any User Task**
> At the beginning of EVERY new user request or task, before modifying any files or writing code, you MUST pull the latest changes from GitHub to ensure the workspace is in sync.

### Required Pre-Task Command:
```bash
git pull origin main
```

- If there are uncommitted local changes that prevent a pull, stash or evaluate local status first (`git status`).
- Resolve any merge conflicts or sync issues before proceeding with code changes.

---

## 2. MANDATORY Post-Phase Guardrail: Commit & Push to GitHub

> [!IMPORTANT]
> **RULE 2: Commit & Push After Every Major Phase of Work**
> Upon completing any major phase, feature implementation, bug fix, or refactoring step:
> 1. Run local verification scripts (`python main.py` or `python demo.py`).
> 2. Ensure virtual environments (`venv/`), `.insightface/`, `__pycache__/`, and `.npy`/`.json` outputs remain ignored.
> 3. Stage, commit with a clear Conventional Commit message, and push to GitHub.

### Required Post-Phase Commands:
```bash
# 1. Check working directory status
git status

# 2. Stage untracked and modified project files
git add .

# 3. Commit with a descriptive Conventional Commit message
git commit -m "<type>(<scope>): <descriptive summary of changes>"

# 4. Push to remote main branch
git push origin main
```

---

## 3. Standard Commit Message Convention

All commit messages MUST follow the Conventional Commit specification:

- `feat:` New features or functionality (e.g., `feat(arcface): add CUDA execution provider support`)
- `fix:` Bug fixes or error resolutions (e.g., `fix(vector): fix dimension mismatch check in compare_vectors`)
- `docs:` Documentation and context updates (e.g., `docs(skills): add workflow guardrails skill`)
- `refactor:` Code restructuring without changing external behavior (e.g., `refactor(cli): modularize main command parser`)
- `test:` Adding or updating test/demo scripts (e.g., `test(demo): add synthetic multi-face test cases`)
- `chore:` Dependency, environment, or configuration updates (e.g., `chore(deps): update requirements.txt with onnxruntime`)

---

## 4. Code Integrity & Environment Verification Guardrails

1. **Never Declare Success Without Verification**: Always execute `.\venv\Scripts\python.exe demo.py` or `.\venv\Scripts\python.exe main.py` after code modifications to verify clean execution.
2. **Preserve .gitignore Rules**: Never force-add virtual environment binaries (`venv/`), model weights (`.insightface/`), or temporary generated output files into Git tracking.
3. **Handle Network & Dependency Failures Gracefully**: When installing pip dependencies, use extended timeouts (`--timeout 1000`) and verify package installations before declaring environment readiness.
