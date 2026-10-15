import os
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, OrderedDict
from flask import Flask, render_template, request, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from engine.processor import process_pdf_files
from engine.retrievers import (
    build_and_save_retriever,
    load_saved_retriever,
    delete_saved_retriever,
)
from engine.llm_handler import get_llm_response, check_hallucination
from engine.db import (
    init_db,
    create_upload_job,
    update_upload_job,
    get_upload_job,
    save_session_metadata,
    get_session_metadata,
    record_query,
    clear_session_data
)

load_dotenv()
flask_secret_key = os.getenv("FLASK_SECRET_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
if not flask_secret_key or flask_secret_key == "replace-with-a-random-secret":
    raise RuntimeError("FLASK_SECRET_KEY must be set to a unique, random value.")
if not groq_api_key or groq_api_key == "your_groq_api_key_here":
    raise RuntimeError("GROQ_API_KEY must be set before starting the application.")
if not google_api_key or google_api_key == "your_google_api_key_here":
    raise RuntimeError("GOOGLE_API_KEY must be set for Gemini document embeddings.")

init_db()

app = Flask(__name__)
app.secret_key = flask_secret_key

# Rate Limiter Configuration (Protects Groq API & server resources)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB max total upload size
_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-upload")

# In-memory retriever cache to avoid repeated disk reads during active sessions
_MAX_ACTIVE_RETRIEVERS = 1
_ACTIVE_RETRIEVERS = OrderedDict()
_RETRIEVER_CACHE_LOCK = threading.Lock()


def _cache_retriever(sess_id: str, retriever):
    """Insert a retriever and evict the least recently used entry if needed."""
    with _RETRIEVER_CACHE_LOCK:
        _ACTIVE_RETRIEVERS[sess_id] = retriever
        _ACTIVE_RETRIEVERS.move_to_end(sess_id)
        while len(_ACTIVE_RETRIEVERS) > _MAX_ACTIVE_RETRIEVERS:
            _ACTIVE_RETRIEVERS.popitem(last=False)


def _get_or_create_session_id():
    """Ensure user has a persistent session cookie."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def _get_active_retriever(sess_id: str):
    """Get retriever from memory cache or load from disk persistence."""
    with _RETRIEVER_CACHE_LOCK:
        if sess_id in _ACTIVE_RETRIEVERS:
            _ACTIVE_RETRIEVERS.move_to_end(sess_id)
            return _ACTIVE_RETRIEVERS[sess_id]
    
    saved_retriever = load_saved_retriever(sess_id)
    if saved_retriever:
        with _RETRIEVER_CACHE_LOCK:
            cached_retriever = _ACTIVE_RETRIEVERS.get(sess_id)
            if cached_retriever is not None:
                _ACTIVE_RETRIEVERS.move_to_end(sess_id)
                return cached_retriever
            _ACTIVE_RETRIEVERS[sess_id] = saved_retriever
            _ACTIVE_RETRIEVERS.move_to_end(sess_id)
            while len(_ACTIVE_RETRIEVERS) > _MAX_ACTIVE_RETRIEVERS:
                _ACTIVE_RETRIEVERS.popitem(last=False)
        return saved_retriever
    
    return None


def _process_upload_job(job_id: str, sess_id: str, saved_paths: list[str], file_names: list[str]):
    """Extract, index, and persist an upload outside the request cycle."""
    job_started_at = time.monotonic()
    print(f"[UPLOAD job={job_id} session={sess_id}] Started with {len(saved_paths)} file(s).", flush=True)
    update_upload_job(job_id, "processing")
    try:
        extraction_started_at = time.monotonic()
        print(f"[UPLOAD job={job_id} session={sess_id}] Starting PDF extraction.", flush=True)
        new_chunks = process_pdf_files(saved_paths)
        print(
            f"[UPLOAD job={job_id} session={sess_id}] PDF extraction completed in "
            f"{time.monotonic() - extraction_started_at:.1f}s with {len(new_chunks)} chunk(s).",
            flush=True,
        )
        if not new_chunks:
            raise ValueError("Could not extract readable text from the uploaded PDFs")

        print(f"[UPLOAD job={job_id} session={sess_id}] Loading existing session index.", flush=True)
        existing_meta = get_session_metadata(sess_id)
        existing_files = existing_meta.get("file_names", [])
        existing_retriever = _get_active_retriever(sess_id)
        all_chunks = (
            existing_retriever.chunks + new_chunks
            if existing_retriever
            else new_chunks
        )
        if len(all_chunks) > 1000:
            raise ValueError("Session exceeds the 1000-chunk limit; clear documents before uploading more.")
        all_files = list(dict.fromkeys(existing_files + file_names))

        print(
            f"[UPLOAD job={job_id} session={sess_id}] Starting index build for "
            f"{len(all_chunks)} total chunk(s).",
            flush=True,
        )
        retriever = build_and_save_retriever(all_chunks, sess_id, new_chunks=new_chunks)
        _cache_retriever(sess_id, retriever)
        print(f"[UPLOAD job={job_id} session={sess_id}] Saving session metadata.", flush=True)
        save_session_metadata(sess_id, all_files, len(all_chunks))

        paper_counts = dict(Counter([doc.metadata.get("source", "Unknown") for doc in all_chunks]))
        update_upload_job(
            job_id,
            "completed",
            result={
                "message": f"Successfully indexed {len(new_chunks)} chunks from {len(file_names)} file(s).",
                "files": all_files,
                "total_chunks": len(all_chunks),
                "paper_chunks": paper_counts,
            },
        )
        print(
            f"[UPLOAD job={job_id} session={sess_id}] Completed in "
            f"{time.monotonic() - job_started_at:.1f}s.",
            flush=True,
        )
    except Exception as e:
        print(f"[ERROR] [UPLOAD job={job_id} session={sess_id}] {type(e).__name__}: {e}", flush=True)
        update_upload_job(job_id, "failed", error=f"Failed to process PDFs: {e}")
    finally:
        _remove_saved_uploads(saved_paths)


def _remove_saved_uploads(saved_paths: list[str]):
    for path in saved_paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"[WARN] Could not remove temporary upload {path}: {e}")


@app.route("/")
def landing():
    """Render landing page."""
    return render_template("landing.html")


@app.route("/workspace")
def workspace():
    """Render main research console."""
    _get_or_create_session_id()
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
@limiter.limit("15 per hour")  # Abuse prevention: max 15 uploads per hour per user
def upload_files():
    """Accept PDF uploads and queue indexing outside the request cycle."""
    sess_id = _get_or_create_session_id()

    if "files" not in request.files:
        return jsonify({"error": "No files attached in request"}), 400

    uploaded_files = request.files.getlist("files")
    if not uploaded_files or uploaded_files[0].filename == "":
        return jsonify({"error": "No files selected"}), 400

    saved_paths = []
    file_names = []

    for file in uploaded_files:
        if file and file.filename.lower().endswith(".pdf"):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            unique_name = f"{sess_id}_{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(save_path)
            saved_paths.append(save_path)

    if not saved_paths:
        return jsonify({"error": "No valid PDF files were uploaded"}), 400

    job_id = uuid.uuid4().hex
    try:
        create_upload_job(job_id, sess_id)
        _UPLOAD_EXECUTOR.submit(_process_upload_job, job_id, sess_id, saved_paths, file_names)
    except Exception as e:
        update_upload_job(job_id, "failed", error=f"Could not queue upload: {e}")
        _remove_saved_uploads(saved_paths)
        raise

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/upload/<job_id>", methods=["GET"])
@limiter.exempt
def upload_status(job_id: str):
    """Return the state of an upload owned by the current session."""
    sess_id = _get_or_create_session_id()
    job = get_upload_job(job_id, sess_id)
    if job is None:
        return jsonify({"error": "Upload job not found"}), 404
    return jsonify(job)


@app.route("/api/chat", methods=["POST"])
@limiter.limit("30 per minute")  # Rate limit: max 30 queries per minute per user
def chat():
    """Grounded query answering using persistent hybrid index and Groq."""
    sess_id = _get_or_create_session_id()
    retriever = _get_active_retriever(sess_id)

    if not retriever:
        return jsonify({"error": "No active documents found. Please upload research papers first."}), 400

    data = request.get_json() or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    start_time = time.time()

    try:
        # Retrieve and rerank top candidates
        docs = retriever.invoke(query)

        if not docs:
            return jsonify({
                "answer": "No relevant sections could be found in the uploaded documents to answer this question.",
                "confidence": 0.0,
                "sources": [],
                "latency": round(time.time() - start_time, 2)
            })

        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "snippet": doc.page_content[:240] + "..."
            }
            for doc in docs
        ]

        # Fast Groq Cloud Generation (Qwen 27B)
        answer, confidence = get_llm_response(query, context)

        # Hallucination Verification for lower-confidence answers
        is_hallucinated = False
        if confidence < 0.75:
            is_hallucinated = check_hallucination(answer, context)
            if is_hallucinated:
                answer = (
                    "**Ungrounded Claim Suppressed:** The system verified that an assertion was generated "
                    "which could not be definitively supported by the uploaded papers."
                )
                confidence = 0.0

        latency = round(time.time() - start_time, 2)
        unique_source_names = list({s["source"] for s in sources})

        # Record to persistent SQLite database
        record_query(sess_id, query, answer, confidence, latency, unique_source_names)

        return jsonify({
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "unique_sources": unique_source_names,
            "latency": latency,
            "hallucination_detected": is_hallucinated
        })

    except Exception as e:
        print(f"[ERROR] Chat error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Retrieve session status, checking disk persistence if necessary."""
    sess_id = _get_or_create_session_id()
    meta = get_session_metadata(sess_id)
    
    retriever = _get_active_retriever(sess_id)
    paper_counts = {}
    if retriever:
        paper_counts = dict(Counter([doc.metadata.get("source", "Unknown") for doc in retriever.chunks]))

    return jsonify({
        "files": meta.get("file_names", []),
        "total_chunks": meta.get("total_chunks", 0),
        "paper_chunks": paper_counts
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear session data from memory, disk, and database."""
    sess_id = _get_or_create_session_id()
    with _RETRIEVER_CACHE_LOCK:
        _ACTIVE_RETRIEVERS.pop(sess_id, None)

    delete_saved_retriever(sess_id)
    clear_session_data(sess_id)
    return jsonify({"success": True, "message": "Corpus and index successfully cleared."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
