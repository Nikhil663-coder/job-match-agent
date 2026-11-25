from __future__ import annotations

import math
import tempfile
from pathlib import Path

from app.agent import JobMatchAgent
from app.db import JobDatabase
from app.models import CandidateProfile, EvalCaseResult, EvalSummary, JobCreate
from app.search import EmbeddingProvider, HybridJobSearcher, job_to_text
from scripts.seed_jobs import SAMPLE_JOBS


GOLDEN_EVAL_CASES = [
    {
        "case_id": "llm_agents",
        "query": "LLM agents tool calling structured outputs Python",
        "expected_titles": ["Software Engineering - AI Agents"],
        "profile": CandidateProfile(skills=["Python", "LLMs", "SQL"], preferred_locations=["Gurgaon"]),
    },
    {
        "case_id": "rag_evals",
        "query": "RAG pipeline evaluation retrieval ranking embeddings",
        "expected_titles": ["ML Engineer - Search & RAG", "Software Engineering - AI Agents"],
        "profile": CandidateProfile(skills=["Python", "Embeddings", "FastAPI"]),
    },
    {
        "case_id": "backend_sql",
        "query": "FastAPI backend SQL APIs Docker microservices",
        "expected_titles": ["Backend Engineer - Python & SQL"],
        "profile": CandidateProfile(skills=["Python", "FastAPI", "SQL", "Docker"]),
    },
]


def _recall_at_k(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    expected = set(expected_titles)
    hits = sum(1 for t in retrieved_titles if t in expected)
    return hits / max(len(expected), 1)


def _mrr(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    expected = set(expected_titles)
    for idx, title in enumerate(retrieved_titles, start=1):
        if title in expected:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    expected = set(expected_titles)
    dcg = 0.0
    for idx, title in enumerate(retrieved_titles, start=1):
        rel = 1.0 if title in expected else 0.0
        if rel > 0:
            dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(expected), len(retrieved_titles))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def run_eval_suite(k: int = 3) -> EvalSummary:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = JobDatabase(str(Path(tmpdir) / "eval.db"))
        embedder = EmbeddingProvider(api_key="")  # deterministic offline eval
        for raw in SAMPLE_JOBS:
            job = JobCreate(**raw)
            db.create_job(job, embedding=embedder.embed(job_to_text(raw)))

        searcher = HybridJobSearcher(db=db, embedder=embedder)
        agent = JobMatchAgent(db=db, searcher=searcher)

        results: list[EvalCaseResult] = []
        for case in GOLDEN_EVAL_CASES:
            hits, _ = searcher.search(case["query"], top_k=k, profile=case["profile"])
            titles = [h.job.title for h in hits]
            reply = agent.chat(
                session_id=f"eval-{case['case_id']}",
                message=case["query"],
                profile=case["profile"],
            )
            structured_ok = bool(reply.summary and reply.recommendations and reply.tools_used)

            results.append(
                EvalCaseResult(
                    case_id=case["case_id"],
                    query=case["query"],
                    recall_at_k=round(_recall_at_k(titles, case["expected_titles"]), 4),
                    mrr=round(_mrr(titles, case["expected_titles"]), 4),
                    ndcg_at_k=round(_ndcg_at_k(titles, case["expected_titles"]), 4),
                    structured_valid=structured_ok,
                )
            )

        n = len(results)
        return EvalSummary(
            num_cases=n,
            mean_recall_at_k=round(sum(r.recall_at_k for r in results) / n, 4),
            mean_mrr=round(sum(r.mrr for r in results) / n, 4),
            mean_ndcg_at_k=round(sum(r.ndcg_at_k for r in results) / n, 4),
            structured_pass_rate=round(sum(1 for r in results if r.structured_valid) / n, 4),
            cases=results,
        )


if __name__ == "__main__":
    summary = run_eval_suite()
    print(summary.model_dump_json(indent=2))
