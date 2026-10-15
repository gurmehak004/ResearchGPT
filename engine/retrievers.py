import os
import re
import shutil
import pickle
import threading
import time
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from engine.storage import DATA_DIR

_embeddings = None
_ranker = None
_embeddings_lock = threading.Lock()
_ranker_lock = threading.Lock()

FAISS_RETRIEVER_K = 8
BM25_RETRIEVER_K = 5
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_RETRIES = 2

INDEX_STORAGE_DIR = os.path.join(DATA_DIR, "indices")
os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)


class GeminiRetrievalEmbeddings(Embeddings):
    """Use Gemini's hosted retrieval embeddings instead of local PyTorch."""

    def __init__(self):
        self.client = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    def _embed_with_retry(self, texts: list[str], task_type: str) -> list[list[float]]:
        for attempt in range(EMBEDDING_MAX_RETRIES + 1):
            try:
                return self.client.embed_documents(
                    texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSION,
                )
            except Exception as e:
                message = str(e)
                is_rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not is_rate_limited or attempt == EMBEDDING_MAX_RETRIES:
                    raise

                retry_match = re.search(r"retry in ([0-9.]+)s", message, re.IGNORECASE)
                delay = float(retry_match.group(1)) if retry_match else 5.0 * (2 ** attempt)
                delay = min(max(delay, 1.0), 90.0)
                print(
                    f"[EMBEDDINGS] Gemini rate limit reached; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{EMBEDDING_MAX_RETRIES}).",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("Gemini embedding retries were exhausted")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_with_retry(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_retry([text], "RETRIEVAL_QUERY")[0]


def _get_embeddings():
    """Lazily initialize the hosted embeddings client once."""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = GeminiRetrievalEmbeddings()
    return _embeddings


def _get_ranker():
    """Lazy-load ultra-lightweight CPU reranker."""
    global _ranker
    if os.getenv("DISABLE_RERANKER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    if _ranker is None:
        with _ranker_lock:
            if _ranker is None:
                try:
                    from flashrank import Ranker

                    _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=".cache/flashrank")
                except ImportError:
                    return None
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
                from flashrank import RerankRequest

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


def _build_faiss_index(
    chunks: list[Document],
    embeddings,
    session_id: str,
    existing_vectorstore=None,
):
    stage_started_at = time.monotonic()
    if existing_vectorstore is not None:
        print(
            f"[INDEX session={session_id}] Adding {len(chunks)} new chunk(s) "
            f"to the existing FAISS index.",
            flush=True,
        )
        vectorstore = existing_vectorstore
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
            vectorstore.add_documents(batch)
            del batch
    else:
        print(
            f"[INDEX session={session_id}] Starting batched FAISS embedding/index build "
            f"for {len(chunks)} chunk(s), batch size {EMBEDDING_BATCH_SIZE}.",
            flush=True,
        )
        vectorstore = None
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            del batch

    if vectorstore is None:
        raise ValueError("Cannot build an index without document chunks")

    print(
        f"[INDEX session={session_id}] FAISS update completed in "
        f"{time.monotonic() - stage_started_at:.1f}s.",
        flush=True,
    )
    return vectorstore


def _load_existing_faiss(session_dir: str, embeddings, session_id: str):
    index_path = os.path.join(session_dir, "index.faiss")
    if not os.path.exists(index_path):
        return None

    vectorstore = FAISS.load_local(
        session_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    if vectorstore.index.d != EMBEDDING_DIMENSION:
        print(
            f"[WARN] [INDEX session={session_id}] Existing index has "
            f"{vectorstore.index.d} dimensions; rebuilding with {EMBEDDING_DIMENSION}.",
            flush=True,
        )
        return None
    return vectorstore


def build_and_save_retriever(
    chunks: list[Document],
    session_id: str,
    new_chunks: list[Document] | None = None,
):
    """
    Build/update FAISS and BM25, embedding only newly uploaded chunks when possible.
    """
    if not chunks:
        raise ValueError("Cannot build retriever from empty chunks list")

    stage_started_at = time.monotonic()
    print(
        f"[INDEX session={session_id}] Initializing embeddings for "
        f"{len(new_chunks or chunks)} new chunk(s).",
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

    has_existing_index = os.path.exists(os.path.join(session_dir, "index.faiss"))
    vectorstore = _load_existing_faiss(session_dir, embeddings, session_id)
    if vectorstore is not None:
        if new_chunks is None:
            raise ValueError("New chunks are required to update an existing FAISS index")
        additions = new_chunks
    else:
        additions = chunks
        if has_existing_index:
            print(
                f"[INDEX session={session_id}] Rebuilding the full FAISS index because "
                "the saved index is incompatible.",
                flush=True,
            )
    vectorstore = _build_faiss_index(additions, embeddings, session_id, vectorstore)

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
        with open(chunks_file, "rb") as f:
            data = pickle.load(f)
            chunks = data["chunks"]
            bm25 = data["bm25"]
            bm25.k = BM25_RETRIEVER_K

        vectorstore = _load_existing_faiss(session_dir, embeddings, session_id)
        if vectorstore is None:
            vectorstore = _build_faiss_index(chunks, embeddings, session_id)
            vectorstore.save_local(session_dir)

        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": FAISS_RETRIEVER_K})

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