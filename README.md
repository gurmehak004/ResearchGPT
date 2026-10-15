Title: ResearchGPT: Hybrid Retrieval-Augmented Generation.

The Problem: Solving hallucinations in academic research.

The Tech Stack: Llama 3.2, FAISS (Dense Retrieval), BM25 (Sparse Retrieval), LangChain, and Streamlit.

Key Features: * Recursive Character Text Splitting.

Local LLM inference for data privacy.

Source-backed citations.

## Render production configuration

Set `GROQ_API_KEY` and a unique, random `FLASK_SECRET_KEY` in the Render service
environment. Set `DISABLE_RERANKER=true` to keep FlashRank disabled.

Indexes and the SQLite database must be stored on a Render persistent disk if
they need to survive deploys and restarts. Attach a disk mounted at `/var/data`
and set `RESEARCHGPT_DATA_DIR=/var/data`. Run a single service instance when
using this disk-backed SQLite configuration.
