<<<<<<< HEAD
AI.LLM — Quick start (Poetry)
================================

This README explains how to set up and run the AI.LLM project locally using Poetry, on Linux and Windows (WSL recommended). It covers environment creation, installing required libraries, handling large artifacts (models/embeds), running the retriever API, and common troubleshooting notes.

Prerequisites
-------------
- Git 2.30+
- Python 3.10/3.11 (3.12 may work but check compatibility)
- Poetry (recommended) — https://python-poetry.org/docs/#installation
- Optional (recommended): Conda / Mamba for system-level packages (faiss, some binary wheels)
- For Windows: WSL2 is strongly recommended. Native Windows support for faiss and some ML wheels may be problematic.

Quick repo notes
----------------
- Large artifacts (models, embed_data, PL_dialogue, out_lora_adapter, etc.) are NOT tracked in git. See `LARGE_ARTIFACTS_COPY_INSTRUCTIONS.md` for instructions where to obtain them and how to restore locally.
- A clean branch `Do/llm-api-clean` exists on the remote that excludes large artifacts. Use it if you just want to run the code without model blobs.

1) Create a project environment with Poetry
-----------------------------------------
Poetry manages a project virtual environment. We recommend using Poetry to create and activate an env, then installing the requirements from `requirements.txt` (this is simpler than manually adding many packages to pyproject).

Linux / macOS (recommended):

```bash
# install poetry if not installed (follow official instructions if you prefer)
curl -sSL https://install.python-poetry.org | python3 -
# open a new shell or ensure `poetry` is on PATH
poetry --version

# Create and enter a Poetry shell (this will create a venv)
cd /path/to/AI.LLM
poetry shell

# Inside the poetry shell, install runtime dependencies from requirements.txt
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Windows (native)
-----------------
- Option A: Use WSL2 and the Linux steps above (recommended).
- Option B: If you must run natively, install Poetry (PowerShell) and use a conda environment for heavy packages: some packages (faiss, torch with CUDA) are easier to install via conda.\\

Example (PowerShell + Conda):\\
```powershell
conda create -n ai-llm python=3.11 -y
conda activate ai-llm
pip install poetry
poetry shell
python -m pip install -r requirements.txt
```

Notes about PyTorch / FAISS / GPU
---------------------------------
- PyTorch: choose the correct wheel for your CUDA version. Visit https://pytorch.org/get-started/locally and copy the recommended `pip` command. Example (Linux CPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu` or simply `pip install torch` (may install cpu build).
- FAISS: `faiss-cpu` on Linux is usually available via pip, but on Windows it may not be. If faiss fails to install, use conda:
	- `conda install -c conda-forge faiss-cpu`
- If you cannot install faiss, the code falls back partly to BM25 retrieval (rank-bm25). Some features require faiss for semantic search.

Installing optional GPU tooling (NVIDIA)
----------------------------------------
- Make sure NVIDIA driver + CUDA toolkit are installed. Then install torch with CUDA support as recommended on pytorch.org.

2) Environment variables (.env)
-------------------------------
Create a `.env` at project root (not checked into git). The project reads environment variables in various places. Example `.env` (adjust):

```dotenv
# LLM provider settings
LLM_PROVIDER=local-hf
LLM_MODEL_ID=models/phogpt4b-ft-merged-fix
LLM_TEMP=0.2
LLM_MAX_NEW=1024
LLM_ENABLE_MEMORY=0

# HuggingFace offline flags (set if you want offline behavior)
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# Ollama (if used)
LLM_OLLAMA_HOST=http://localhost:11434

# Other runtime
PORT=8001
```

3) Restoring large artifacts (models / embed_data)
-------------------------------------------------
- See `LARGE_ARTIFACTS_COPY_INSTRUCTIONS.md` in the repository for instructions where to get them and how to copy them into the workspace.\\

Put the downloaded artifacts under the repo root as:
- `models/` (model files, tokenizer, etc.)
- `embed_data/` (embedding data JSONs)
- `out_lora_adapter/` (adapter weights)
- `PL_dialogue/PLGTDB_emb.json` (if required)

Do NOT add these files to git. Use Git LFS or external artifact stores for sharing if needed.

4) Running the retriever API (local)
-----------------------------------
The main FastAPI-based retriever app is `retriever.api_server:app`. To run in development mode:

```bash
# From repo root
# Use uvicorn inside poetry environment
python -m uvicorn retriever.api_server:app --host 0.0.0.0 --port 8001 --reload

# or
uvicorn retriever.api_server:app --host 0.0.0.0 --port 8001 --reload
```

Example curl query after server starts:\
```bash
curl -X POST "http://127.0.0.1:8001/query" -H "Content-Type: application/json" -d '{"question":"vượt xe trong hầm đường bộ có được phép hay không"}' | jq
```

5) Running tests (if available)
-------------------------------
The project includes tests under `tests/`. To run them (pytest assumed):\\

```bash
# install pytest if necessary
python -m pip install pytest
pytest -q
```

6) Development tips & common issues
----------------------------------
- Git LFS: if you plan to keep large artifacts in the repo, install Git LFS and track the file types: `git lfs install; git lfs track "*.safetensors"`.
- Push failures (HTTP 408 / 500) usually mean the repo contains very large objects. Remove them from history (BFG/git-filter-repo) or use Git LFS.
- If `uvicorn` is not found, either `python -m uvicorn ...` or install uvicorn inside environment: `pip install uvicorn`.
- On Windows, prefer WSL2. Native Windows sometimes fails to install `faiss-cpu` or some binary wheels.
- If you see warnings about `.git/hooks` not executable, it’s safe; you can `chmod +x .git/hooks/pre-push` if you want to use hooks locally.

7) Minimal troubleshooting checklist
-----------------------------------
- Server won't start: check `python -m pip list` to ensure `fastapi`, `uvicorn` are installed; check `PYTHONPATH` if you run from outside repo root.
- LLM model not found: ensure `LLM_MODEL_ID` points to local path (e.g. `models/...`) or remote identifier; set `HF_HUB_OFFLINE=0` if you want HF network access.
- FAISS import error: install via conda-forge or use CPU-only pip wheel.
- Push to remote failing with large files: run `git count-objects -vH` and `git rev-list --objects --all | sort` to find large blobs.

Appendix: Useful commands summary
--------------------------------
# Create & enter Poetry env
poetry shell

# Install dependencies (inside poetry environment)
python -m pip install -r requirements.txt

# Start retriever API
python -m uvicorn retriever.api_server:app --host 0.0.0.0 --port 8001 --reload

# Run a quick curl test
curl -X POST "http://127.0.0.1:8001/query" -H "Content-Type: application/json" -d '{"question":"đạo đức là gì"}' | jq

# If faiss fails on Windows: consider WSL/Conda
conda install -c conda-forge faiss-cpu

Final notes
-----------
If you want, I can:
- Produce a `poetry.lock` / `pyproject.toml` populated from `requirements.txt` (automated approach) or convert the requirements to Poetry format.
- Create a `scripts/restore_artifacts.sh` helper to automate copying artifacts from a given path.
- Help migrate the repository history to Git LFS or run BFG to purge previously committed model files (destructive).\\

Tell me which next step you'd like and I will implement it (create pyproject conversion, restore script, or run history purge with your confirmation).
=======
# AI.LLM
chatbot CORE
>>>>>>> origin/main
