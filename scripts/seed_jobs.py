from __future__ import annotations

from app.db import JobDatabase
from app.models import JobCreate
from app.search import EmbeddingProvider, job_to_text

SAMPLE_JOBS = [
    {
        "title": "Software Engineering - AI Agents",
        "company": "Elip AI",
        "location": "Gurgaon",
        "remote": False,
        "min_ctc_lpa": 25.0,
        "skills": ["Python", "LLMs", "Embeddings", "SQL", "FastAPI", "Docker"],
        "description": (
            "Build LLM-powered agents with tool use, memory, and structured outputs. "
            "Run semantic search and personalized ranking over a constantly changing jobs index, "
            "and evaluate LLM output quality in production."
        ),
    },
    {
        "title": "ML Engineer - Search & RAG",
        "company": "VectorScale",
        "location": "Bengaluru",
        "remote": True,
        "min_ctc_lpa": 22.0,
        "skills": ["Python", "Embeddings", "RAG", "Vector DB", "Evals"],
        "description": (
            "Design hybrid BM25 and dense vector retrieval pipelines, two-stage cross-encoder "
            "reranking, and automated RAG evaluation harnesses."
        ),
    },
    {
        "title": "Backend Engineer - Python & SQL",
        "company": "FinGrid",
        "location": "Mumbai",
        "remote": False,
        "min_ctc_lpa": 18.0,
        "skills": ["Python", "FastAPI", "SQL", "PostgreSQL", "Docker"],
        "description": (
            "Build reliable REST APIs, optimize SQL queries and indexing, and deploy containerized "
            "Python microservices."
        ),
    },
    {
        "title": "Applied AI Engineer",
        "company": "CopilotLabs",
        "location": "Hyderabad",
        "remote": True,
        "min_ctc_lpa": 20.0,
        "skills": ["Python", "Prompting", "Tool Calling", "Pydantic", "APIs"],
        "description": (
            "Ship customer-facing LLM workflows using function calling, JSON schema validation, "
            "and prompt regression testing."
        ),
    },
]


def seed() -> int:
    db = JobDatabase()
    if db.list_jobs():
        return 0
    embedder = EmbeddingProvider()
    for raw in SAMPLE_JOBS:
        payload = JobCreate(**raw)
        db.create_job(payload, embedding=embedder.embed(job_to_text(raw)))
    return len(SAMPLE_JOBS)


if __name__ == "__main__":
    count = seed()
    print(f"Seeded {count} jobs.")
