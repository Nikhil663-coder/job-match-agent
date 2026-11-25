from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.db import JobDatabase
from app.models import (
    AgentStructuredReply,
    CandidateProfile,
    RecommendedJob,
)
from app.search import HybridJobSearcher

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search the live jobs index using hybrid semantic + keyword retrieval and candidate personalization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Role, stack, or domain query"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_skill_gap",
            "description": "Compare candidate skills against a specific job posting ID to find missing skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Target job ID"},
                    "candidate_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skills the candidate already has",
                    },
                },
                "required": ["job_id", "candidate_skills"],
            },
        },
    },
]


class JobMatchAgent:
    """Tool-using career assistant with conversation memory and structured Pydantic outputs."""

    def __init__(self, db: JobDatabase, searcher: HybridJobSearcher) -> None:
        self.db = db
        self.searcher = searcher
        self._client = None
        if settings.openai_api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)

    def _run_tool(
        self, name: str, args: dict[str, Any], profile: CandidateProfile | None
    ) -> dict[str, Any]:
        if name == "search_jobs":
            hits, total = self.searcher.search(
                query=args["query"],
                top_k=int(args.get("top_k", 3)),
                profile=profile,
            )
            return {
                "total_indexed": total,
                "results": [
                    {
                        "job_id": h.job.id,
                        "title": h.job.title,
                        "company": h.job.company,
                        "location": h.job.location,
                        "remote": h.job.remote,
                        "min_ctc_lpa": h.job.min_ctc_lpa,
                        "skills": h.job.skills,
                        "matched_skills": h.matched_skills,
                        "score": h.final_score,
                    }
                    for h in hits
                ],
            }

        if name == "analyze_skill_gap":
            job = self.db.get_job(int(args["job_id"]))
            if not job:
                return {"error": f"Job {args['job_id']} not found"}
            have = {s.lower() for s in args.get("candidate_skills", [])}
            missing = [s for s in job.skills if s.lower() not in have]
            return {"job_id": job.id, "title": job.title, "missing_skills": missing}

        return {"error": f"Unknown tool: {name}"}

    def chat(
        self,
        session_id: str,
        message: str,
        profile: CandidateProfile | None = None,
    ) -> AgentStructuredReply:
        history = self.db.get_recent_messages(session_id, limit=6)
        self.db.append_message(session_id, "user", message)

        if self._client is not None:
            reply = self._chat_openai(message, history, profile)
        else:
            reply = self._chat_local(message, history, profile)

        self.db.append_message(session_id, "assistant", reply.summary)
        return reply

    def _chat_openai(
        self,
        message: str,
        history: list[dict[str, str]],
        profile: CandidateProfile | None,
    ) -> AgentStructuredReply:
        assert self._client is not None
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a concise AI job-matching assistant. Use the provided tools to search "
                    "the live jobs index and analyze skill gaps. Ground every recommendation in tool output."
                ),
            },
            *history,
            {
                "role": "user",
                "content": (
                    f"{message}\n\nCandidate profile: {profile.model_dump_json() if profile else 'none'}"
                ),
            },
        ]

        tools_used: list[str] = []
        for _ in range(3):
            resp = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break

            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")
                tools_used.append(fn_name)
                tool_out = self._run_tool(fn_name, fn_args, profile)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_out),
                    }
                )

        parsed = self._client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=messages,
            response_format=AgentStructuredReply,
        )
        result = parsed.choices[0].message.parsed
        if result is None:
            return self._chat_local(message, history, profile)
        result.tools_used = list(dict.fromkeys(tools_used))
        return result

    def _chat_local(
        self,
        message: str,
        history: list[dict[str, str]],
        profile: CandidateProfile | None,
    ) -> AgentStructuredReply:
        # Combine recent user context with current query so follow-ups remember prior turns
        prior_user_text = " ".join(m["content"] for m in history if m["role"] == "user")
        effective_query = f"{prior_user_text} {message}".strip()

        search_out = self._run_tool("search_jobs", {"query": effective_query, "top_k": 3}, profile)
        tools_used = ["search_jobs"]
        results = search_out.get("results", [])

        recs: list[RecommendedJob] = []
        missing_all: list[str] = []
        candidate_skills = profile.skills if profile else []

        for item in results:
            matched = item.get("matched_skills", [])
            reason = (
                f"Matches {', '.join(matched)}"
                if matched
                else f"Relevant to query ({item['location']}, {item['min_ctc_lpa']} LPA)"
            )
            recs.append(
                RecommendedJob(
                    job_id=item["job_id"],
                    title=item["title"],
                    company=item["company"],
                    reason=reason,
                    match_score=item["score"],
                )
            )

        if results and candidate_skills:
            gap = self._run_tool(
                "analyze_skill_gap",
                {"job_id": results[0]["job_id"], "candidate_skills": candidate_skills},
                profile,
            )
            tools_used.append("analyze_skill_gap")
            missing_all = gap.get("missing_skills", [])

        intent = "skill_gap" if "missing" in message.lower() or "gap" in message.lower() else "search"
        if not recs:
            summary = "No matching roles found in the current index."
        else:
            top = recs[0]
            summary = f"Found {len(recs)} matching roles. Top match: {top.title} at {top.company} (score {top.match_score})."

        return AgentStructuredReply(
            summary=summary,
            intent=intent,
            recommendations=recs,
            missing_skills=missing_all,
            tools_used=tools_used,
        )
