from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from app.config import settings
from app.db import JobDatabase
from app.models import CandidateProfile, JobRecord, SearchHit

LOCAL_DIM = 128
_TOKEN_RE = re.compile(r"[a-z0-9\+\#\.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def job_to_text(job: JobRecord | dict) -> str:
    if isinstance(job, dict):
        skills = ", ".join(job.get("skills", []))
        return (
            f"{job['title']} at {job['company']} ({job['location']}). "
            f"Skills: {skills}. {job['description']}"
        )
    skills = ", ".join(job.skills)
    return f"{job.title} at {job.company} ({job.location}). Skills: {skills}. {job.description}"


class EmbeddingProvider:
    """Uses OpenAI embeddings when configured, otherwise a deterministic local hashing projection."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self._client = None
        if self.api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        if self._client is not None:
            resp = self._client.embeddings.create(model=self.model, input=text)
            return list(resp.data[0].embedding)
        return self._local_embed(text)

    @staticmethod
    def _local_embed(text: str) -> list[float]:
        vec = np.zeros(LOCAL_DIM, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec.tolist()
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % LOCAL_DIM
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.shape != vb.shape:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return max(0.0, float(np.dot(va, vb) / denom))


def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized_docs = [_tokenize(d) for d in docs]
    query_terms = _tokenize(query)
    n_docs = len(tokenized_docs)
    if n_docs == 0 or not query_terms:
        return [0.0] * n_docs

    avgdl = sum(len(d) for d in tokenized_docs) / max(n_docs, 1)
    df: Counter[str] = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] += 1

    raw_scores: list[float] = []
    for doc in tokenized_docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_terms:
            if tf[term] == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            num = tf[term] * (k1 + 1.0)
            den = tf[term] + k1 * (1.0 - b + b * (dl / max(avgdl, 1.0)))
            score += idf * (num / den)
        raw_scores.append(score)

    max_score = max(raw_scores, default=0.0)
    if max_score <= 0.0:
        return [0.0] * n_docs
    return [s / max_score for s in raw_scores]


def _personalization_score(
    job: JobRecord, profile: CandidateProfile | None
) -> tuple[float, list[str]]:
    if profile is None:
        return 0.0, []

    job_skills_map = {s.lower(): s for s in job.skills}
    candidate_skills = {s.lower() for s in profile.skills}
    matched = [job_skills_map[s] for s in candidate_skills if s in job_skills_map]

    skill_ratio = len(matched) / max(len(job_skills_map), 1)

    loc_bonus = 0.0
    if profile.remote_only:
        loc_bonus = 0.25 if job.remote else -0.25
    elif profile.preferred_locations:
        prefs = {loc.lower() for loc in profile.preferred_locations}
        if job.location.lower() in prefs or job.remote:
            loc_bonus = 0.20

    ctc_bonus = 0.15 if (profile.min_ctc_lpa <= 0 or job.min_ctc_lpa >= profile.min_ctc_lpa) else -0.20

    score = max(0.0, min(1.0, 0.65 * skill_ratio + 0.20 + loc_bonus + ctc_bonus))
    return score, matched


class HybridJobSearcher:
    """Two-stage retrieval: hybrid semantic+BM25 recall followed by personalized re-ranking."""

    def __init__(self, db: JobDatabase, embedder: EmbeddingProvider | None = None) -> None:
        self.db = db
        self.embedder = embedder or EmbeddingProvider()

    def search(
        self,
        query: str,
        top_k: int = 5,
        profile: CandidateProfile | None = None,
    ) -> tuple[list[SearchHit], int]:
        rows = self.db.list_jobs_with_embeddings()
        if not rows:
            return [], 0

        query_vec = self.embedder.embed(query)
        docs = [job_to_text(job) for job, _ in rows]
        lexical_scores = _bm25_scores(query, docs)

        hits: list[SearchHit] = []
        for idx, (job, emb) in enumerate(rows):
            if emb is None or len(emb) != len(query_vec):
                emb = self.embedder.embed(docs[idx])
                self.db.save_embedding(job.id, emb)

            sem_score = _cosine_similarity(query_vec, emb)
            lex_score = lexical_scores[idx]
            pers_score, matched = _personalization_score(job, profile)

            base_retrieval = 0.55 * sem_score + 0.45 * lex_score
            final_score = (
                0.70 * base_retrieval + 0.30 * pers_score if profile is not None else base_retrieval
            )

            hits.append(
                SearchHit(
                    job=job,
                    semantic_score=round(sem_score, 4),
                    lexical_score=round(lex_score, 4),
                    personalization_score=round(pers_score, 4),
                    final_score=round(final_score, 4),
                    matched_skills=matched,
                )
            )

        hits.sort(key=lambda h: h.final_score, reverse=True)
        return hits[:top_k], len(rows)
