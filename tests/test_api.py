from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.evals import run_eval_suite
from app.main import create_app


class JobMatchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "test.db")
        self.client = TestClient(create_app(db_path=self.db_path))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_crud_search_and_agent_flow(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        # Create two jobs
        r1 = self.client.post(
            "/jobs",
            json={
                "title": "AI Agent Engineer",
                "company": "Elip AI",
                "location": "Gurgaon",
                "remote": False,
                "min_ctc_lpa": 25.0,
                "skills": ["Python", "LLMs", "SQL", "Embeddings"],
                "description": "Build tool-using LLM agents and semantic search over jobs.",
            },
        )
        self.assertEqual(r1.status_code, 201)
        job_id = r1.json()["id"]

        r2 = self.client.post(
            "/jobs",
            json={
                "title": "Frontend React Developer",
                "company": "WebCraft",
                "location": "Pune",
                "remote": True,
                "min_ctc_lpa": 12.0,
                "skills": ["TypeScript", "React", "CSS"],
                "description": "Build accessible web dashboards and component libraries.",
            },
        )
        self.assertEqual(r2.status_code, 201)

        # Search with personalization
        search_resp = self.client.post(
            "/search",
            json={
                "query": "LLM agents semantic search Python",
                "top_k": 2,
                "profile": {
                    "skills": ["Python", "LLMs"],
                    "preferred_locations": ["Gurgaon"],
                    "min_ctc_lpa": 20.0,
                },
            },
        )
        self.assertEqual(search_resp.status_code, 200)
        hits = search_resp.json()["hits"]
        self.assertEqual(hits[0]["job"]["title"], "AI Agent Engineer")
        self.assertIn("Python", hits[0]["matched_skills"])

        # Agent chat with memory + structured output
        chat_1 = self.client.post(
            "/agent/chat",
            json={
                "session_id": "s1",
                "message": "Find LLM agent roles in Gurgaon",
                "profile": {"skills": ["Python", "LLMs"]},
            },
        )
        self.assertEqual(chat_1.status_code, 200)
        body = chat_1.json()
        self.assertIn("search_jobs", body["tools_used"])
        self.assertIn("analyze_skill_gap", body["tools_used"])
        self.assertIn("SQL", body["missing_skills"])

        # Incremental update & delete
        upd = self.client.patch(f"/jobs/{job_id}", json={"min_ctc_lpa": 26.0})
        self.assertEqual(upd.status_code, 200)
        self.assertEqual(upd.json()["min_ctc_lpa"], 26.0)

        delete_resp = self.client.delete(f"/jobs/{job_id}")
        self.assertEqual(delete_resp.status_code, 204)

    def test_eval_suite(self) -> None:
        summary = run_eval_suite(k=3)
        self.assertEqual(summary.num_cases, 3)
        self.assertGreaterEqual(summary.mean_recall_at_k, 0.8)
        self.assertEqual(summary.structured_pass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
