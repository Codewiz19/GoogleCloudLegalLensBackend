"""
rag_service.py - Vertex AI RAG helper functions for FastAPI app.

Exports:
- init_vertex(creds_path: Optional[str])
- create_or_get_corpus(display_name: str)
- import_files_to_corpus(rag_corpus, gs_paths: List[str], chunk_size=512, chunk_overlap=100)
- retrieval_query_simple(rag_corpus_name: str, text: str, top_k: int)
- check_rag_retrieval(rag_corpus_name: str, query_text: str, top_k: int)
- generate_summary_with_tool(rag_corpus_name: str, prompt: str, model_name: str)
- generate_summary_with_tool_and_check(rag_corpus_name: str, prompt: str, model_name: str)
- generate_direct_with_model(prompt: str, model_name: str)
"""

import os
import time
from typing import List, Optional, Tuple, Any, Dict

# google auth
from google.oauth2 import service_account

# Robust imports for vertexai + rag + generative_models
try:
    import vertexai
    from vertexai import rag
except Exception as e:
    raise ImportError(
        "Failed to import `vertexai` or `vertexai.rag`. "
        "Install/upgrade the package `vertexai` and `google-cloud-aiplatform`. "
        "See README. Original error: " + str(e)
    )

try:
    from vertexai.generative_models import GenerativeModel, Tool
except Exception as e:
    raise ImportError(
        "Failed to import GenerativeModel/Tool from vertexai.generative_models. "
        "Ensure you have a compatible `vertexai` SDK version. Original error: " + str(e)
    )


def init_vertex(creds_path: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Initialize vertexai with optional explicit service account JSON.
    Returns a dict with project/location/staging_bucket for convenience.
    """
    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-east4")
    staging_bucket = os.environ.get("STAGING_BUCKET")
    service_account_email = os.environ.get("SERVICE_ACCOUNT_EMAIL")

    creds = None
    if creds_path:
        if not os.path.isfile(creds_path):
            raise FileNotFoundError(f"Credentials file not found: {creds_path}")
        creds = service_account.Credentials.from_service_account_file(creds_path).with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )

    # init vertex
    vertexai.init(project=project, location=location, staging_bucket=staging_bucket, credentials=creds, service_account=service_account_email)
    return {"project": project, "location": location, "staging_bucket": staging_bucket}


def create_or_get_corpus(display_name: str):
    """
    Create a new RAG corpus. For MVP we create a new corpus with provided display name.
    (You can extend to reuse existing corpora by storing the name externally.)
    """
    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )
    rag_corpus = rag.create_corpus(
        display_name=display_name,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=embedding_model_config
        ),
    )
    return rag_corpus


def import_files_to_corpus(rag_corpus, gs_paths: List[str], chunk_size: int = 512, chunk_overlap: int = 100) -> str:
    """
    Import files into the given rag_corpus (list of gs:// paths).
    This triggers an asynchronous import on Vertex side.
    Returns the corpus name (string).
    """
    rag.import_files(
        rag_corpus.name,
        gs_paths,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ),
        max_embedding_requests_per_min=1000,
    )
    return rag_corpus.name


def retrieval_query_simple(rag_corpus_name: str, text: str, top_k: int = 3) -> Any:
    """
    Simple retrieval query wrapper returning the raw SDK response. Useful for debugging.
    """
    rag_retrieval_config = rag.RagRetrievalConfig(top_k=top_k)
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=rag_corpus_name)],
        text=text,
        rag_retrieval_config=rag_retrieval_config,
    )
    return response


def check_rag_retrieval(rag_corpus_name: str, query_text: str = "test retrieval", top_k: int = 3) -> Any:
    """
    Run a retrieval query and return the raw response or error dict.
    This is used by the app to quickly determine whether any indexed passages exist.
    """
    try:
        resp = retrieval_query_simple(rag_corpus_name, query_text, top_k=top_k)
        return resp
    except Exception as e:
        return {"error": str(e)}


def generate_summary_with_tool(rag_corpus_name: str, prompt: str, model_name: str = "gemini-2.0-flash-001") -> str:
    """
    Create a retrieval tool from the Vertex RAG corpus and generate content using the tool.
    Returns the string text (attempting .text first then fallback to string).
    """
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=rag_corpus_name)],
                rag_retrieval_config=rag.RagRetrievalConfig(top_k=4),
            ),
        )
    )
    rag_model = GenerativeModel(model_name=model_name, tools=[rag_retrieval_tool])
    gen_response = rag_model.generate_content(prompt)
    try:
        return gen_response.text
    except Exception:
        return str(gen_response)


def _heuristic_retrieval_has_results(raw_retrieval: Any) -> bool:
    """
    Heuristics to decide if a raw retrieval response contains useful passages.
    - If retrieval is a dict with 'error' key -> False
    - If stringified response contains keywords like 'documents' or long content -> True
    This is intentionally conservative (prefers False unless we're confident).
    """
    try:
        s = str(raw_retrieval).lower()
        if not s:
            return False
        if "error" in s:
            return False
        # common markers
        if "documents" in s or "content" in s or "text" in s or len(s) > 120:
            return True
        return False
    except Exception:
        return False


def generate_summary_with_tool_and_check(rag_corpus_name: str, prompt: str, model_name: str = "gemini-2.0-flash-001") -> Tuple[Optional[str], Dict[str, Any]]:
    """
    1) Runs a lightweight retrieval check.
    2) If retrieval returns results, creates a retrieval tool + generator and returns the generated text.
    3) Returns (summary_text_or_None, debug_info_dict)
    """
    debug: Dict[str, Any] = {"rag_corpus": rag_corpus_name}
    raw_retrieval = check_rag_retrieval(rag_corpus_name, query_text="Check: return any passages", top_k=4)
    debug["retrieval_raw"] = str(raw_retrieval)

    if _heuristic_retrieval_has_results(raw_retrieval):
        try:
            # Build retrieval tool and run generation
            rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=rag_corpus_name)],
                        rag_retrieval_config=rag.RagRetrievalConfig(top_k=4),
                    ),
                )
            )
            rag_model = GenerativeModel(model_name=model_name, tools=[rag_retrieval_tool])
            gen_response = rag_model.generate_content(prompt)
            try:
                text = gen_response.text
            except Exception:
                text = str(gen_response)
            debug["generation_used_retrieval"] = True
            return text, debug
        except Exception as e:
            debug["generation_error"] = str(e)
            return None, debug
    else:
        # No retrieval results yet
        debug["generation_used_retrieval"] = False
        return None, debug


def generate_direct_with_model(prompt: str, model_name: str = "gemini-2.0-flash-001") -> str:
    """
    Call the generative model directly (no retrieval tool). Useful as fallback.
    """
    rag_model = GenerativeModel(model_name=model_name)
    gen_response = rag_model.generate_content(prompt)
    try:
        return gen_response.text
    except Exception:
        return str(gen_response)
