from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.config import settings
from app.models import JobCreate, JobRecord, JobUpdate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobDatabase:
    """SQLite store for job postings, cached embeddings, and chat memory."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    location TEXT NOT NULL,
                    remote INTEGER NOT NULL DEFAULT 0,
                    min_ctc_lpa REAL NOT NULL DEFAULT 0.0,
                    skills_json TEXT NOT NULL,
                    description TEXT NOT NULL,
                    embedding_json TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_memory(session_id, id)"
            )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            title=row["title"],
            company=row["company"],
            location=row["location"],
            remote=bool(row["remote"]),
            min_ctc_lpa=float(row["min_ctc_lpa"]),
            skills=json.loads(row["skills_json"]),
            description=row["description"],
            updated_at=row["updated_at"],
        )

    def create_job(self, payload: JobCreate, embedding: list[float] | None = None) -> JobRecord:
        now = _utc_now()
        emb_json = json.dumps(embedding) if embedding is not None else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs (title, company, location, remote, min_ctc_lpa, skills_json, description, embedding_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.title,
                    payload.company,
                    payload.location,
                    int(payload.remote),
                    payload.min_ctc_lpa,
                    json.dumps(payload.skills),
                    payload.description,
                    emb_json,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._row_to_job(row)

    def get_job(self, job_id: int) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY id ASC").fetchall()
        return [self._row_to_job(r) for r in rows]

    def list_jobs_with_embeddings(self) -> list[tuple[JobRecord, list[float] | None]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY id ASC").fetchall()
        result: list[tuple[JobRecord, list[float] | None]] = []
        for row in rows:
            emb = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            result.append((self._row_to_job(row), emb))
        return result

    def update_job(
        self,
        job_id: int,
        payload: JobUpdate,
        embedding: list[float] | None = None,
    ) -> JobRecord | None:
        existing = self.get_job(job_id)
        if not existing:
            return None

        merged = existing.model_copy(
            update={k: v for k, v in payload.model_dump(exclude_none=True).items()}
        )
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET title = ?, company = ?, location = ?, remote = ?, min_ctc_lpa = ?,
                    skills_json = ?, description = ?, embedding_json = COALESCE(?, embedding_json), updated_at = ?
                WHERE id = ?
                """,
                (
                    merged.title,
                    merged.company,
                    merged.location,
                    int(merged.remote),
                    merged.min_ctc_lpa,
                    json.dumps(merged.skills),
                    merged.description,
                    json.dumps(embedding) if embedding is not None else None,
                    now,
                    job_id,
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row)

    def delete_job(self, job_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cur.rowcount > 0

    def save_embedding(self, job_id: int, embedding: list[float]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET embedding_json = ? WHERE id = ?",
                (json.dumps(embedding), job_id),
            )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_memory (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, _utc_now()),
            )

    def get_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM chat_memory
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
