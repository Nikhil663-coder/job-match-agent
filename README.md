# Job Match Agent

Semantic job search, personalized ranking, and a tool-using career assistant built with FastAPI, SQLite, and structured LLM outputs.

## What It Does

- **Dynamic Jobs Index (SQLite):** Create, update, and delete job postings with incremental index refreshes.
- **Hybrid Search & Re-ranking:** Combines lexical matching and vector embeddings (OpenAI `text-embedding-3-small` with an offline deterministic fallback) followed by personalized re-ranking on candidate skills, location, and compensation.
- **Tool-Using Agent:** Maintains per-session conversation memory, invokes search and profile tools via JSON schemas, and returns validated Pydantic structured responses.
- **Evaluation Harness:** Measures retrieval quality (`Recall@K`, `MRR`, `NDCG@K`) and structured-output/constraint adherence against a golden test set.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: enable live OpenAI embeddings & chat completions
cp .env.example .env

python -m scripts.seed_jobs
uvicorn app.main:app --reload
```

API docs are available at `http://127.0.0.1:8000/docs`.

## Running Tests & Evals

```bash
python -m unittest discover -s tests -v
python -m app.evals
```

## Docker

```bash
docker build -t job-match-agent .
docker run -p 8000:8000 --env-file .env job-match-agent
```
