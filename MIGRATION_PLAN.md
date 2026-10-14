# 🚀 ResearchGPT: Migration Plan to Flask + Groq End-to-End Application

## 1. Objective
Transition **ResearchGPT** from a local-only setup (Chainlit + Ollama) to a **full-stack, production-ready Flask web application** powered by the **Groq API** (Llama 3.3 70B), complete with a shareable public deployment configuration.

---

## 2. Key Architecture Differences

| Feature | Current Implementation | New Implementation |
| :--- | :--- | :--- |
| **LLM Provider** | Local Ollama (`llama3.2:3b`) | **Groq Cloud API** (`llama-3.3-70b-versatile`) |
| **Compute Requirement** | Heavy local RAM/CPU/GPU | **Lightweight** (can run on free cloud servers like Render) |
| **Web Framework** | Chainlit | **Flask + REST API + Vanilla HTML/CSS/JS UI** |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | HuggingFace `all-MiniLM-L6-v2` (optimized) |
| **Retrieval** | FAISS + BM25 Ensemble | FAISS + BM25 Ensemble (per-session isolated) |
| **Deployment** | Localhost only | **Public URL** (Render / Railway / HuggingFace Spaces) |

---

## 3. Implementation Steps

1. **Step 1: Environment & Dependencies**
   - Update `requirements.txt` to include `flask`, `groq`, `python-dotenv`, `gunicorn`.
   - Create `.env.example` with `GROQ_API_KEY` placeholder.

2. **Step 2: LLM Engine (`engine/llm_handler.py`)**
   - Replace `ollama` SDK with official `groq` SDK.
   - Use `llama-3.3-70b-versatile` with JSON mode response format.
   - Maintain grounded answering, confidence score calculation, and optional hallucination verification.

3. **Step 3: Flask Backend (`app.py`)**
   - Session management (UUID-based or Flask session) to isolate user PDF indexes.
   - Endpoints:
     - `GET /` — Main UI.
     - `POST /api/upload` — Multipart PDF upload, chunking, and index build.
     - `POST /api/chat` — Context retrieval, Groq generation, confidence, and source attribution.
     - `GET /api/status` — Status check on loaded documents.
     - `POST /api/reset` — Clear loaded session index.

4. **Step 4: Frontend Development**
   - `templates/index.html`: Semantic, responsive layout with side panel (documents loaded, metrics) and main chat window.
   - `static/css/style.css`: Clean, modern dark/light aesthetic, glassmorphic accents, animations.
   - `static/js/app.js`: Dynamic file drop-and-upload, chat UI rendering, markdown rendering, citations expander.

5. **Step 5: Cloud Deployment Configuration**
   - Create `Procfile` for Render / Railway (`web: gunicorn app:app`).
   - Add deployment instructions.
