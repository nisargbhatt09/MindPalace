# MindPalace POC — Quickstart

A local web app that remembers your day from your photos and answers questions
like *"where was I in the morning?"* — grounded only in your real photos.

Architecture, rationale, and roadmap live in [`PLAN.md`](./PLAN.md).
Stack: **FastAPI + SQLite (on-device) + Mistral (Pixtral vision & chat)** backend,
**React + Vite + TypeScript** frontend. No BLIP, no Pinecone, no embeddings —
deterministic SQL retrieval over LLM-generated captions + tags.

## 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your MISTRAL_API_KEY
python seed.py                # optional: load a simulated day to chat with
uvicorn app.main:app --reload --port 8000
```

## 2. Frontend

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

## 3. Try it

After seeding (or uploading photos), ask:

- "Where was I in the morning?" → cafe + park
- "Did I see any animals today?" → the dog photo
- "What did I have for lunch?" → the pasta
- "Was I at the beach today?" → *"I don't have a photo from the beach"* — **never invented**

That last one is the make-or-break test: the assistant must refuse to fabricate
a memory. See the `memory-assistant` skill for the full safety checklist.
