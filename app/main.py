from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.agent import JobMatchAgent
from app.db import JobDatabase
from app.evals import run_eval_suite
from app.models import (
    AgentStructuredReply,
    ChatRequest,
    EvalSummary,
    JobCreate,
    JobRecord,
    JobUpdate,
    SearchRequest,
    SearchResponse,
)
from app.search import EmbeddingProvider, HybridJobSearcher, job_to_text


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Job Match Agent", version="0.1.0")
    db = JobDatabase(db_path=db_path)
    embedder = EmbeddingProvider()
    searcher = HybridJobSearcher(db=db, embedder=embedder)
    agent = JobMatchAgent(db=db, searcher=searcher)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return db.list_jobs()

    @app.post("/jobs", response_model=JobRecord, status_code=201)
    def create_job(payload: JobCreate) -> JobRecord:
        emb = embedder.embed(job_to_text(payload.model_dump()))
        return db.create_job(payload, embedding=emb)

    @app.patch("/jobs/{job_id}", response_model=JobRecord)
    def update_job(job_id: int, payload: JobUpdate) -> JobRecord:
        existing = db.get_job(job_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Job not found")
        merged = existing.model_copy(
            update={k: v for k, v in payload.model_dump(exclude_none=True).items()}
        )
        emb = embedder.embed(job_to_text(merged))
        updated = db.update_job(job_id, payload, embedding=emb)
        assert updated is not None
        return updated

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: int) -> None:
        if not db.delete_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")

    @app.post("/search", response_model=SearchResponse)
    def search_jobs(payload: SearchRequest) -> SearchResponse:
        hits, total = searcher.search(
            query=payload.query,
            top_k=payload.top_k,
            profile=payload.profile,
        )
        return SearchResponse(query=payload.query, total_indexed=total, hits=hits)

    @app.post("/agent/chat", response_model=AgentStructuredReply)
    def agent_chat(payload: ChatRequest) -> AgentStructuredReply:
        return agent.chat(
            session_id=payload.session_id,
            message=payload.message,
            profile=payload.profile,
        )

    @app.post("/evals/run", response_model=EvalSummary)
    def run_evals() -> EvalSummary:
        return run_eval_suite()

    return app


app = create_app()
