import os
import shutil
import pickle
import threading
import time
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from engine.storage import DATA_DIR

try:
    from flashrank import Ranker, RerankRequest
    _flashrank_available = True
except ImportError:
    _flashrank_available = False

_embeddings = None
_ranker = None
_embeddings_lock = threading.Lock()
_ranker_lock = threading.Lock()

FAISS_RETRIEVER_K = 8
BM25_RETRIEVER_K = 5

INDEX_STORAGE_DIR = os.path.join(DATA_DIR, "indices")
os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)


def _get_embeddings():
    """Lazy-load embeddings model once."""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"batch_size": 128, "normalize_embeddings": False}
                )
    return _embeddings


def _get_ranker():
    """Lazy-load ultra-lightweight CPU reranker."""
    global _ranker
    if os.getenv("DISABLE_RERANKER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    if _ranker is None and _flashrank_available:
        with _ranker_lock:
            if _ranker is None:
                try:
                    _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=".cache/flashrank")
                except Exception as e:
                    print(f"[WARN] Failed to load FlashRank: {e}")
                    _ranker = None
    return _ranker


class PersistentResearchRetriever:
    """
    Two-stage hybrid retriever supporting disk persistence and reloads.
    """

    def __init__(self, faiss_retriever, bm25_retriever, chunks: list[Document]):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.chunks = chunks
        self.abstract_chunks = [
            doc for doc in chunks if doc.metadata.get("is_abstract_chunk", False)
        ]

    def invoke(self, query: str) -> list[Document]:
        query_lower = query.lower()
        is_overview_query = any(k in query_lower for k in [
            "abstract", "summary", "summarize", "overview", "what is this paper about", "intro", "introduction"
        ])

        candidates = []
        try:
            dense_docs = self.faiss_retriever.invoke(query)
            candidates.extend(dense_docs)
        except Exception as e:
            print(f"[WARN] FAISS retrieval failed: {e}")

        try:
            sparse_docs = self.bm25_retriever.invoke(query)
            candidates.extend(sparse_docs)
        except Exception as e:
            print(f"[WARN] BM25 retrieval failed: {e}")

        if is_overview_query and self.abstract_chunks:
            candidates = self.abstract_chunks + candidates

        seen = set()
        unique_candidates = []
        for doc in candidates:
            sample = doc.page_content[:150]
            if sample not in seen:
                seen.add(sample)
                unique_candidates.append(doc)

        if not unique_candidates:
            return []

        ranker = _get_ranker()
        if ranker and len(unique_candidates) > 1:
            try:
                passages = [
                    {"id": idx, "text": doc.page_content, "meta": doc.metadata}
                    for idx, doc in enumerate(unique_candidates)
                ]
                rerank_request = RerankRequest(query=query, passages=passages)
                results = ranker.rerank(rerank_request)

                top_results = results[:6]
                return [
                    Document(page_content=r["text"], metadata=r["meta"])
                    for r in top_results
                ]
            except Exception as e:
                print(f"[WARN] Rerank fallback: {e}")

        return unique_candidates[:6]


def build_and_save_retriever(chunks: list[Document], session_id: str):
    """
    Build FAISS + BM25 and persist both directly to disk under session_id.
    """
    if not chunks:
        raise ValueError("Cannot build retriever from empty chunks list")

    stage_started_at = time.monotonic()
    print(
        f"[INDEX session={session_id}] Initializing embeddings for {len(chunks)} chunk(s).",
        flush=True,
    )
    embeddings = _get_embeddings()
    print(
        f"[INDEX session={session_id}] Embeddings initialized in "
        f"{time.monotonic() - stage_started_at:.1f}s.",
        flush=True,
    )
    session_dir = os.path.join(INDEX_STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    stage_started_at = time.monotonic()
    print(
        f"[INDEX session={session_id}] Starting FAISS embedding/index build for {len(chunks)} chunk(s).",
        flush=True,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(
        f"[INDEX session={session_id}] FAISS embedding/index build completed in "
        f"{time.monotonic() - stage_started_at:.1f}s.",
        flush=True,
    )

    stage_started_at = time.monotonic()
    print(f"[INDEX session={session_id}] Saving FAISS index.", flush=True)
    vectorstore.save_local(session_dir)
    print(
        f"[INDEX session={session_id}] FAISS index saved in "
        f"{time.monotonic() - stage_started_at:.1f}s.",
        flush=True,
    )

    stage_started_at = time.monotonic()
    print(f"[INDEX session={session_id}] Starting BM25 build.", flush=True)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = BM25_RETRIEVER_K
    print(
        f"[INDEX session={session_id}] BM25 build completed in "
        f"{time.monotonic() - stage_started_at:.1f}s.",
        flush=True,
    )

    stage_started_at = time.monotonic()
    print(f"[INDEX session={session_id}] Saving BM25 data and chunks.", flush=True)
    with open(os.path.join(session_dir, "bm25_chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "bm25": bm25}, f)
    print(
        f"[INDEX session={session_id}] BM25 data saved in "
        f"{time.monotonic() - stage_started_at:.1f}s.",
        flush=True,
    )

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": FAISS_RETRIEVER_K})
    return PersistentResearchRetriever(faiss_retriever, bm25, chunks)


def load_saved_retriever(session_id: str):
    """
    Load persisted FAISS + BM25 index from disk without re-parsing PDFs.
    """
    session_dir = os.path.join(INDEX_STORAGE_DIR, session_id)
    faiss_index_file = os.path.join(session_dir, "index.faiss")
    chunks_file = os.path.join(session_dir, "bm25_chunks.pkl")

    if not (os.path.exists(faiss_index_file) and os.path.exists(chunks_file)):
        return None

    try:
        embeddings = _get_embeddings()
        vectorstore = FAISS.load_local(session_dir, embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": FAISS_RETRIEVER_K})

        with open(chunks_file, "rb") as f:
            data = pickle.load(f)
            chunks = data["chunks"]
            bm25 = data["bm25"]
            bm25.k = BM25_RETRIEVER_K

        print(f"[INFO] Successfully loaded persistent index from disk for session {session_id}")
        return PersistentResearchRetriever(faiss_retriever, bm25, chunks)
    except Exception as e:
        print(f"[WARN] Failed to load saved retriever: {e}")
        return None


def delete_saved_retriever(session_id: str):
    """Delete index folder on session reset."""
    session_dir = os.path.join(INDEX_STORAGE_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)