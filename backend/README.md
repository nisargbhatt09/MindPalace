# MindPalace backend

FastAPI service: ingest photos (EXIF + Claude vision description) and answer
grounded natural-language questions about the person's day.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Optionally seed a simulated day so you can chat without uploading photos:

```bash
python seed.py
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | liveness + active model |
| GET  | `/api/memories` | list stored memories |
| POST | `/api/ingest` | upload an image (multipart `file`) → caption + tags + metadata |
| GET  | `/api/memories/{id}/image` | fetch a stored image |
| POST | `/api/chat` | `{message, history}` → grounded `{answer, sources, found_memories}` |

## How it works

- **Deterministic:** EXIF time/GPS extraction, optional reverse geocoding, SQL
  time/place filtering, content ranking over caption + tags (`repository.py`).
- **LLM:** photo description + semantic tags at ingest (`vision.py`); turning a
  question into `search_memories` calls and writing the grounded reply
  (`assistant.py`).
- **Storage:** on-device SQLite (`db.py`). The repository layer keeps a future
  swap to Postgres + pgvector isolated to one module.

The assistant is instructed never to fabricate a memory: empty retrieval → it
says it has no photo from then.
