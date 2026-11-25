from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=160)
    company: str = Field(..., min_length=1, max_length=120)
    location: str = Field(..., min_length=1, max_length=120)
    remote: bool = False
    min_ctc_lpa: float = Field(default=0.0, ge=0.0)
    skills: list[str] = Field(default_factory=list)
    description: str = Field(..., min_length=10)


class JobUpdate(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    remote: bool | None = None
    min_ctc_lpa: float | None = Field(default=None, ge=0.0)
    skills: list[str] | None = None
    description: str | None = None


class JobRecord(JobCreate):
    id: int
    updated_at: str


class CandidateProfile(BaseModel):
    skills: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    remote_only: bool = False
    min_ctc_lpa: float = Field(default=0.0, ge=0.0)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    profile: CandidateProfile | None = None


class SearchHit(BaseModel):
    job: JobRecord
    semantic_score: float
    lexical_score: float
    personalization_score: float
    final_score: float
    matched_skills: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    total_indexed: int
    hits: list[SearchHit]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1)
    profile: CandidateProfile | None = None


class RecommendedJob(BaseModel):
    job_id: int
    title: str
    company: str
    reason: str
    match_score: float


class AgentStructuredReply(BaseModel):
    summary: str
    intent: Literal["search", "skill_gap", "clarify", "general"]
    recommendations: list[RecommendedJob] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class EvalCaseResult(BaseModel):
    case_id: str
    query: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    structured_valid: bool


class EvalSummary(BaseModel):
    num_cases: int
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    structured_pass_rate: float
    cases: list[EvalCaseResult]
