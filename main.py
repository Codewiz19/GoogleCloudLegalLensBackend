"""
main.py - corrected FastAPI app for Vertex AI RAG MVP

Endpoints:
- POST /upload_pdf    : upload PDF -> extract text -> upload to GCS -> returns doc_id & gs_path
- POST /summarize     : try RAG-grounded summary (poll retrieval); fallback to direct generation
- POST /risks         : deterministic server-side risk extraction + LLM elaboration (fallback deterministic)
- POST /chat          : RAG chat (try retrieval-grounded; fallback to direct generation)
- GET  /debug_rag/{doc_id} : debug helper to inspect retrieval raw output
"""
import os
import uuid
import tempfile
import time
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Project helpers (ensure these modules exist in your project)
from gcs_utils import upload_file_to_gcs
from pdf_utils import extract_text_from_pdf
from rag_service import (
    init_vertex,
    create_or_get_corpus,
    import_files_to_corpus,
    generate_summary_with_tool_and_check,
    generate_direct_with_model,
    check_rag_retrieval,
)
from prompts import SUMMARIZE_PROMPT, RISKS_PROMPT, CHAT_SYSTEM_PROMPT
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Legal RAG FastAPI (Vertex AI) - Corrected")

@app.get("/")
async def root():
    return {"message": "RAG Legal Document Analysis API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "API is running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store (for MVP). Persist in DB for production.
DOC_STORE: Dict[str, Dict[str, Any]] = {}
CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}

# Initialize Vertex (best-effort). We intentionally do not crash app if init fails.
try:
    init_vertex(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    print("Vertex initialized (or attempted).")
except Exception as e:
    print("Vertex init failed or skipped:", e)


class UploadResponse(BaseModel):
    doc_id: str
    gs_path: str
    message: str
    corpus_name: Optional[str] = None


@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    # Validate PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    contents = await file.read()
    tmp.write(contents)
    tmp.flush()
    tmp.close()

    # Extract text (page-level offsets included)
    try:
        text_info = extract_text_from_pdf(tmp.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")

    # Upload to GCS
    bucket_url = os.environ.get("STAGING_BUCKET")
    if not bucket_url:
        raise HTTPException(status_code=500, detail="STAGING_BUCKET not configured (set env var).")

    bucket_name = bucket_url.replace("gs://", "").strip("/")
    dest_name = f"uploads/{uuid.uuid4()}_{file.filename}"

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and not os.path.isfile(creds_path):
        raise HTTPException(status_code=500, detail=(f"GOOGLE_APPLICATION_CREDENTIALS is set to '{creds_path}', but the file was not found."))

    try:
        gs_path = upload_file_to_gcs(tmp.name, bucket_name, dest_name, creds_path=creds_path)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=500, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF to GCS: {e}")

    # Create a corpus immediately (best-effort); we store corpus_name so later calls reuse it.
    corpus_name = None
    try:
        display = f"legal_doc_{uuid.uuid4().hex[:8]}"
        rag_corpus = create_or_get_corpus(display)
        corpus_name = rag_corpus.name
    except Exception as e:
        # Non-fatal: keep going. Import will be attempted in /summarize.
        print("Warning: create_or_get_corpus failed at upload:", e)
        corpus_name = None

    doc_id = str(uuid.uuid4())
    DOC_STORE[doc_id] = {
        "gs_path": gs_path,
        "filename": file.filename,
        "text_info": text_info,
        "corpus_name": corpus_name,
        "uploaded_at": time.time(),
    }

    return UploadResponse(doc_id=doc_id, gs_path=gs_path, message="Uploaded and extracted text.", corpus_name=corpus_name)


class SummarizeRequest(BaseModel):
    doc_id: str
    display_name: Optional[str] = "legal_doc_corpus"


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if req.doc_id not in DOC_STORE:
        raise HTTPException(status_code=404, detail="doc_id not found.")
    doc = DOC_STORE[req.doc_id]
    gs_path = doc["gs_path"]

    # Ensure corpus exists (reuse previously created one if available)
    corpus_name = doc.get("corpus_name")
    rag_corpus = None
    if not corpus_name:
        try:
            rag_corpus = create_or_get_corpus(req.display_name + "_" + req.doc_id[:8])
            corpus_name = rag_corpus.name
            doc["corpus_name"] = corpus_name
        except Exception as e:
            print("Warning: could not create corpus:", e)
            corpus_name = None

    # Import file into corpus (best-effort). This triggers an async import on Vertex side.
    if corpus_name and rag_corpus is None:
        # If we only had name but not object, recreate rag_corpus object using create_or_get_corpus
        try:
            rag_corpus = create_or_get_corpus(req.display_name + "_" + req.doc_id[:8])
        except Exception:
            rag_corpus = None

    if rag_corpus:
        try:
            import_files_to_corpus(rag_corpus, [gs_path])
        except Exception as e:
            print("Warning: import_files_to_corpus threw:", e)

    # Try to generate RAG-grounded summary by polling retrieval/generation helper.
    # generate_summary_with_tool_and_check returns (summary_text_or_None, debug_dict)
    max_tries = 8
    wait_seconds = 15
    last_debug = None
    for attempt in range(max_tries):
        if corpus_name:
            try:
                summary, debug = generate_summary_with_tool_and_check(corpus_name, SUMMARIZE_PROMPT + "\n\nSummarize the document and cite passages as [start:end].")
            except Exception as e:
                summary = None
                debug = {"error": str(e)}
        else:
            summary = None
            debug = {"error": "no corpus available"}

        last_debug = debug
        if summary:
            return {"doc_id": req.doc_id, "summary": summary, "rag_corpus": corpus_name, "debug": debug, "fallback": False}
        # not ready -> wait and retry
        time.sleep(wait_seconds)

    # After polling, no retrieval results: fallback to direct generation using extracted text
    doc_text = doc["text_info"].get("full_text", "")
    if not doc_text:
        raise HTTPException(status_code=500, detail={"message": "No extracted text available and RAG retrieval returned no data.", "debug": last_debug})

    max_chars = 15000
    doc_excerpt = doc_text[:max_chars]
    fallback_prompt = (
        "Document:\n" + doc_excerpt + "\n\n"
        + SUMMARIZE_PROMPT + "\n\n"
        + "Using the document above, produce a concise summary (3-6 sentences), a bullet list "
        + "of the most important obligations/clauses, and a one-paragraph plain-language explanation."
    )
    try:
        fallback_summary = generate_direct_with_model(fallback_prompt)
        return {"doc_id": req.doc_id, "summary": fallback_summary, "rag_corpus": corpus_name, "debug": last_debug, "fallback": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "RAG retrieval empty and fallback generation failed.", "debug": last_debug, "fallback_error": str(e)})


class RisksRequest(BaseModel):
    doc_id: str


@app.post("/risks")
def risks(req: RisksRequest):
    if req.doc_id not in DOC_STORE:
        raise HTTPException(status_code=404, detail="doc_id not found.")
    text = DOC_STORE[req.doc_id]["text_info"]["full_text"]

    # Deterministic rule-based risk extraction (regex-based)
    import re
    patterns = [
        ("indemnif", "Indemnity / broad indemnify clause", 30),
        ("penalt", "Penalties / liquidated damages", 25),
        ("governing law", "Unfavorable governing law / jurisdiction", 20),
        ("termination", "One-sided termination rights", 20),
        ("warrant", "Broad or missing warranties", 15),
        ("liabilit", "Limitation of liability / unlimited liability", 25),
        ("confident", "Weak data protection / confidentiality", 20),
        ("assign", "Assignment restrictions or transfers", 10),
        ("notice", "Notice periods that are too short", 8),
        ("automatic", "Automatic renewals", 12),
    ]
    found = []
    for idx, (pat, label, weight) in enumerate(patterns):
        for m in re.finditer(pat, text, flags=re.I):
            start = m.start()
            end = m.end()
            snippet = text[max(0, start - 80): min(len(text), end + 80)]
            found.append({
                "id": f"risk-{idx}-{start}",
                "label": label,
                "start": start,
                "end": end,
                "snippet": snippet,
                "score": weight
            })

    # Merge nearby hits and aggregate scores
    found_sorted = sorted(found, key=lambda x: x["start"])
    merged = []
    for item in found_sorted:
        if not merged:
            merged.append(item.copy()); continue
        last = merged[-1]
        if item["start"] - last["end"] <= 200:
            last["end"] = max(last["end"], item["end"])
            last["snippet"] = text[max(0, last["start"] - 80): min(len(text), last["end"] + 80)]
            last["score"] = min(100, last.get("score", 0) + item["score"])
            last["label"] = last["label"] + " ; " + item["label"]
        else:
            merged.append(item.copy())

    for r in merged:
        sc = r["score"]
        r["severity_score"] = sc
        r["severity_level"] = "High" if sc >= 70 else ("Medium" if sc >= 40 else "Low")

    server_provided = [
        {
            "id": r["id"],
            "severity_level": r["severity_level"],
            "severity_score": r["severity_score"],
            "snippet": r["snippet"],
            "label": r["label"],
        }
        for r in merged
    ]

    # Ask LLM to elaborate (deterministic prompt). Prefer direct generation using server_provided JSON.
    prompt = RISKS_PROMPT + "\n\nServerRisks: " + json.dumps(server_provided, ensure_ascii=False)
    try:
        # Try direct generation (deterministic)
        generated = generate_direct_with_model(prompt)
        try:
            parsed = json.loads(generated)
            return {"doc_id": req.doc_id, "risks": parsed}
        except Exception:
            # LLM output not parseable -> return raw with server_provided
            return {"doc_id": req.doc_id, "raw_llm_output": generated, "server_risks": server_provided}
    except Exception as e:
        # Fallback deterministic structured advice
        fallback = []
        for s in server_provided:
            fallback.append({
                "id": s["id"],
                "severity_level": s["severity_level"],
                "severity_score": s["severity_score"],
                "short_risk": s.get("label", "Risk"),
                "explanation": "Detected snippet with keywords; review the clause near the provided snippet.",
                "recommendations": ["Narrow the clause", "Add caps/time limits", "Add explicit data-protection language"]
            })
        return {"doc_id": req.doc_id, "risks": fallback, "note": f"LLM attempt failed: {e}"}


class ChatRequest(BaseModel):
    doc_id: str
    messages: List[Dict[str, str]]  # [{"role":"user","content":"..."}]
    session_id: Optional[str] = None


@app.post("/chat")
def chat(req: ChatRequest):
    if req.doc_id not in DOC_STORE:
        raise HTTPException(status_code=404, detail="doc_id not found.")

    doc = DOC_STORE[req.doc_id]
    corpus_name = doc.get("corpus_name")

    # Ensure corpus exists and import file (best-effort)
    if not corpus_name:
        try:
            rag_corpus = create_or_get_corpus("chat_" + req.doc_id[:8])
            corpus_name = rag_corpus.name
            doc["corpus_name"] = corpus_name
        except Exception as e:
            print("Warning: could not create corpus for chat:", e)
            corpus_name = None

    if corpus_name:
        try:
            import_files_to_corpus(create_or_get_corpus("chat_" + req.doc_id[:8]), [doc["gs_path"]])
        except Exception:
            pass

    # Use last user message as the query
    last_user = req.messages[-1]["content"]
    system_prompt = CHAT_SYSTEM_PROMPT
    prompt = system_prompt + "\n\nConversation: " + last_user

    # Try retrieval-grounded chat first
    if corpus_name:
        try:
            summary, debug = generate_summary_with_tool_and_check(corpus_name, prompt)
            if summary:
                return {"doc_id": req.doc_id, "response": summary, "debug": debug, "fallback": False}
        except Exception as e:
            print("Chat retrieval/generation error:", e)

    # Fallback to direct generation using a doc excerpt
    doc_text = DOC_STORE[req.doc_id]["text_info"].get("full_text", "")
    doc_excerpt = doc_text[:8000]
    fallback_prompt = "Document:\n" + doc_excerpt + "\n\n" + system_prompt + "\n\nConversation: " + last_user
    try:
        response = generate_direct_with_model(fallback_prompt)
        return {"doc_id": req.doc_id, "response": response, "fallback": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat fallback generation failed: {e}")


@app.get("/debug_rag/{doc_id}")
def debug_rag(doc_id: str):
    """
    Debug helper. Returns the raw retrieval response for the corpus associated with doc_id.
    Useful to inspect whether the corpus was indexed and what retrieval returns.
    """
    if doc_id not in DOC_STORE:
        raise HTTPException(status_code=404, detail="doc_id not found.")
    corpus_name = DOC_STORE[doc_id].get("corpus_name")
    if not corpus_name:
        return {"doc_id": doc_id, "error": "No corpus_name recorded for this document."}
    retrieval = check_rag_retrieval(corpus_name, query_text="debug retrieval", top_k=5)
    return {"doc_id": doc_id, "rag_corpus": corpus_name, "retrieval_raw": str(retrieval)}
