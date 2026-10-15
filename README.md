Title: ResearchGPT: Hybrid Retrieval-Augmented Generation.

The Problem: Solving hallucinations in academic research.

The Tech Stack: Llama 3.2, FAISS (Dense Retrieval), BM25 (Sparse Retrieval), LangChain, and Streamlit.

Key Features: * Recursive Character Text Splitting.

Local LLM inference for data privacy.

Source-backed citations.

## Render production configuration

Set `GROQ_API_KEY`, `GOOGLE_API_KEY`, and a unique, random `FLASK_SECRET_KEY`
in the Render service environment. Set `DISABLE_RERANKER=true` to keep
FlashRank disabled. Gemini embedding quotas and pricing depend on your Google
AI Studio account and current plan.

Render persistent disks require a paid service. To keep indexes and the SQLite
database across deploys and restarts, attach a disk mounted at `/var/data` and
set `RESEARCHGPT_DATA_DIR=/var/data`. Run a single service instance when using
this disk-backed SQLite configuration. On the free plan, leave
`RESEARCHGPT_DATA_DIR` unset; files use the service's ephemeral filesystem and
can be lost after restarts or deploys.

PDF uploads are queued and processed asynchronously so indexing does not have
to finish within a single HTTP request. An upload still in progress can be
interrupted if the service restarts.

For the 512 MB free instance, local PyTorch/Sentence Transformers embeddings
have been replaced with Gemini hosted embeddings. BM25 remains enabled. Uploads
are capped at 10 MB total, 100 pages per PDF, and 1000 chunks per session;
FAISS vectors are generated in small batches. Adding another PDF embeds only
its new chunks into FAISS, while BM25 is rebuilt for the full session. Gemini
429 rate-limit responses are retried using the server-provided delay.
