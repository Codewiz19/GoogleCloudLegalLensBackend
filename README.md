# RAG FastAPI (Vertex AI) — Legal Document Summarizer (MVP)

This repository contains an MVP FastAPI backend to build a RAG-based legal document summarizer using **Google Vertex AI RAG Engine** and **Google Cloud Storage (GCS)**.

**Important:** This code is an opinionated, production-friendly starting point. You must provide your own Google service account JSON and set environment variables described below.

## Repo structure

- `main.py` — FastAPI app with endpoints:
  - `POST /upload_pdf` — Upload PDF, store to GCS, extract text.
  - `POST /summarize` — Summarize the whole document using Vertex AI RAG.
  - `POST /risks` — Detect deterministic risks (server-side rules) and get LLM-crafted explanations & remediation.
  - `POST /chat` — RAG chat endpoint for interactive Q&A.
- `rag_service.py` — Wrapper for Vertex AI initialization + RAG operations.
- `gcs_utils.py` — GCS upload/download helpers.
- `pdf_utils.py` — PDF text extraction using PyMuPDF (fitz).
- `prompts.py` — Prompt templates for summarize, risks, and chat.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — Container image for deployment.
- `.env.example` — env variable example.
- `run.sh` — helper to run locally.

## Quick start (local)

1. Create a GCP project, a GCS bucket (staging bucket) and a service account with roles:
   - `Storage Object Admin`
   - `Vertex AI Admin` (or specific RAG roles)
   - `Service Account User` / `IAM Service Account Token Creator`
2. Download the service account JSON and set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa.json"
   export GCP_PROJECT="your-project-id"
   export GCP_LOCATION="us-east4"
   export STAGING_BUCKET="gs://your-staging-bucket"
   export SERVICE_ACCOUNT_EMAIL="your-sa@your-project.iam.gserviceaccount.com"
   ```
3. Create a Python venv and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Notes & Caveats
- Vertex RAG `import_files` can run asynchronously server-side — it may take time to index documents. The server tries a retrieval after import but you should check import status in Vertex Console for large files.
- For deterministic risk *levels*, this project computes deterministic scores server-side (rule/keyword based) and **passes those levels to the LLM** only for generating explanatory text. That makes severity assignment repeatable.
- See Google Vertex AI RAG docs: Quickstart & Import Files for best practices.

