# AGENTS.md — MindPalace

Guidance for AI agents (Claude Code and others) working in this repository.
Read this before making changes.

> **Working on the assistant itself?** Use the `memory-assistant` skill
> (`.claude/skills/memory-assistant/SKILL.md`). It encodes the safe workflow for
> ingest, retrieval, agent tools, prompts, and the chat loop — and enforces the
> no-fabricated-memories rule below.

## What this project is

**MindPalace** is an assistive **memory companion for people living with memory
loss** (Alzheimer's, dementia). The person takes photos through their day; the
system stores each photo with its metadata (when + where) and a description, and
later answers natural-language questions about their own day:

> "Where was I in the morning?" · "Did I see a dog today?" · "What did I have for lunch?"

It answers conversationally, as a gentle assistant — grounded entirely in the
person's real photos.

## ⚠️ The one rule that overrides everything: never fabricate a memory

This app is used by people who rely on it to know what is real. A made-up or
embellished memory can genuinely confuse or distress them.

- The assistant must ground **every** statement in retrieved photo-memories.
- If retrieval returns nothing, it must say so plainly ("I don't have a photo
  from then") and must **not** guess, infer, or fill gaps.
- Every answer should expose which memories it used, so a caregiver can verify.
- When changing prompts, retrieval, or output handling, preserve this guarantee.
  The negative test case ("Was I at the beach today?" → no photo) is the
  make-or-break check for any change.

## Architecture

This started as a CLIP/BLIP image-caption *search* tool and is evolving into a
**metadata-grounded RAG memory assistant**. The key design insight:

> Questions like "where was I in the morning?" are driven by **metadata (when /
> where)**, not visual similarity. Retrieval is **structured-first** (time +
> place filters) with semantic caption search as a secondary content filter.

```
INGEST:  photo → caption (BLIP) → metadata (EXIF time + GPS → place) → store
ASK:     question → LLM picks structured filters → retrieve → grounded answer
```

- **Captioning:** Salesforce BLIP (`blip-image-captioning-large`).
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim).
- **Vector store:** Pinecone (cosine) — see open decisions about local alternatives.
- **Assistant layer:** an LLM with tool calling + validated structured output
  (framework still open — see below).

## Repository layout

```
main.py                          CLI: ingest a directory, or --query to search
mindpalace/
  mindpalace.py                  MindPalace orchestrator (process / search)
  config/settings.py             dataclass config; env via .env
  models/caption_model.py        BLIP image → caption
  models/embedding_model.py      MiniLM text → vector
  database/vector_store.py       Pinecone wrapper (store / query)
test/test.ipynb                  scratch notebook
```

## Setup & commands

```bash
pip install -r requirements.txt

# Ingest images (current behaviour)
python main.py --image-dir ./images

# Semantic search (current behaviour)
python main.py --query "a dog playing in the park" --top-k 3
```

Environment (via `.env`): `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`,
`PINECONE_INDEX_NAME`. The assistant layer will also need an LLM key
(e.g. `MISTRAL_API_KEY`).

## Conventions

- Python 3.10+; type hints on public functions; module + function docstrings
  (match the existing style in `mindpalace/`).
- Small, single-purpose modules. New assistant code goes under
  `mindpalace/assistant/`; new ingest enrichment under `mindpalace/models/`.
- Pydantic models for any data crossing the LLM boundary (validated I/O).
- Add new dependencies to `requirements.txt`.

## POC scope

**Goal:** prove that fuzzy natural-language questions over ~15 photo-memories
get correct, grounded answers — including correctly saying "no photo" when there
is none.

In scope: simulated one-day dataset (time + place + caption), local storage
(JSON / in-memory), structured + semantic retrieval, one LLM with the care
persona, a CLI chat loop, a fixed demo question set with a negative case.

Out of scope (defer): mobile app, camera capture, real-time sync, Pinecone/cloud,
auth, encryption, face recognition, caregiver dashboard, multi-day scale.

## POC stack (decided — implemented in `backend/` + `frontend/`)

The original `mindpalace/` package (BLIP + Pinecone CLI) is **superseded** by the
full-stack POC. Do new work in `backend/` and `frontend/`.

- **Backend:** FastAPI (`backend/app/`). **Frontend:** React + Vite + TS (`frontend/`).
- **Database:** on-device **SQLite** (`backend/app/db.py`), behind a repository
  layer so Postgres + pgvector is a drop-in for production scale.
- **Captioning:** **Mistral Pixtral vision** at ingest → caption + semantic tags
  (`vision.py`). No BLIP.
- **Retrieval:** **deterministic SQL** — time/place filters + tag/caption matching
  (`repository.py`). No embeddings/vector DB at POC scale.
- **Assistant:** **raw Mistral SDK** tool-calling loop (`assistant.py`) — no
  PydanticAI, for auditability of the grounding guarantee.

See `QUICKSTART.md` to run it and `PLAN.md` for the full design.

## Still-open decisions (revisit for production, not the POC)

- Generation could add a **vision-LLM-at-answer-time** path (currently
  caption/metadata grounded) for questions needing visual detail.
- **Storage** privacy/encryption for real health-adjacent data; Postgres+pgvector at scale.
- People/faces ("who was I with?"), reverse-geocoding quality, multi-day scale.
