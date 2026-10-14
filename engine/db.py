import os
import sqlite3
import json
import time

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "researchgpt.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db():
    """Connect to SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables for sessions, documents, and query history."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # User/Session table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL,
                last_active REAL,
                file_names TEXT,
                total_chunks INTEGER DEFAULT 0
            )
        """)

        # Query history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT,
                answer TEXT,
                confidence REAL,
                latency REAL,
                sources TEXT,
                created_at REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        conn.commit()


def save_session_metadata(session_id: str, file_names: list[str], total_chunks: int):
    """Save or update session metadata in SQLite."""
    now = time.time()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, created_at, last_active, file_names, total_chunks)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_active = excluded.last_active,
                file_names = excluded.file_names,
                total_chunks = excluded.total_chunks
        """, (session_id, now, now, json.dumps(file_names), total_chunks))
        conn.commit()


def get_session_metadata(session_id: str) -> dict:
    """Retrieve session metadata from SQLite."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return {"file_names": [], "total_chunks": 0}
        
        return {
            "file_names": json.loads(row["file_names"] or "[]"),
            "total_chunks": row["total_chunks"]
        }


def record_query(session_id: str, query: str, answer: str, confidence: float, latency: float, sources: list):
    """Log a user query to persistent SQLite history."""
    now = time.time()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_history (session_id, query, answer, confidence, latency, sources, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, query, answer, confidence, latency, json.dumps(sources), now))
        conn.commit()


def clear_session_data(session_id: str):
    """Clear session from database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM query_history WHERE session_id = ?", (session_id,))
        conn.commit()
