![MindPalace](https://github.com/user-attachments/assets/cc41da9c-69f5-4644-8100-ec19d5e02094)

# MindPalace

**A gentle memory companion for people living with memory loss.**

MindPalace turns your photos into memories you can ask about in plain words.
Take pictures through your day; MindPalace notes *when* and *where* each one was
taken and writes a warm description. Later, you simply ask —
*"Where was I this morning?"* — and it answers kindly, **grounded only in what
really happened.**

> **The one promise that shapes everything:** MindPalace never invents a memory.
> If there's no photo, it says so — plainly — rather than guess. For someone who
> relies on it to know what's real, a confidently wrong answer is worse than none.

---

## How it works

```
INGEST   photo  →  EXIF (time + GPS)  →  vision model (caption + tags)  →  on-device DB
ASK      "where was I this morning?"  →  the assistant turns this into a time/place
         search  →  deterministic retrieval  →  a warm, grounded answer
```

The core insight: *"where was I in the morning?"* is a **time + place** question,
not an image-similarity one. So retrieval is **deterministic** (SQL filters over
metadata + tags), and the LLM is used where it's genuinely right: describing
photos at ingest, and turning questions into searches + warm replies.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | **React + Vite + TypeScript** | Multi-page product site + the assistant UI |
| Backend | **FastAPI** (Python) | Clean async API |
| Database | **SQLite**, on-device | Local, single file, zero-config; repository layer keeps Postgres + pgvector a drop-in later |
| Vision (ingest) | **Mistral Pixtral** | Caption + broad semantic tags (`dog → dog, animal, pet`) |
| Assistant | **Mistral** (tool-calling) | Question → structured search → grounded answer |
| Retrieval | **Deterministic SQL** | Time/place filters + tag matching; auditable, no hallucination |

> Provider-agnostic by design — swap Mistral for Gemini, OpenAI, NVIDIA NIM, or a
> local Ollama model by changing the client config.

## Quickstart

You need a [Mistral API key](https://console.mistral.ai/). Full walkthrough in
[`QUICKSTART.md`](./QUICKSTART.md).

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your MISTRAL_API_KEY
python seed.py                # optional: load a simulated day to chat with
uvicorn app.main:app --reload --port 8000
```

**Frontend** (new terminal)
```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

Then ask: *"Where was I in the morning?"*, *"Did I see any animals today?"*, or
the make-or-break test — *"Was I at the beach today?"* (it will honestly say it
has no such photo).

## Project structure

```
backend/                 FastAPI + SQLite + Mistral
  app/
    main.py              REST API (health, memories, ingest, image, chat)
    db.py                SQLite schema (on-device)
    repository.py        deterministic retrieval (time/place + tag matching)
    exif.py              EXIF time + GPS extraction
    vision.py            Pixtral → caption + semantic tags
    assistant.py         grounded tool-calling agent + care persona
    ingest.py            bytes → stored image + metadata + description
  seed.py                load a simulated day (chat without uploading)

frontend/                React + Vite + TypeScript
  src/
    pages/               Home, How to use, About, Privacy, the app
    components/          Nav, Footer, Chat, MemoryGallery, Polaroid, Logo
    styles.css           the design system

PLAN.md                  full design, POC scope, roadmap
AGENTS.md                guidance for AI agents working in this repo
QUICKSTART.md            step-by-step run guide
.claude/skills/          the `memory-assistant` build/safety skill
```

## Design philosophy

The interface is built to feel *human* — warm, unhurried, dignified — for an
audience that includes people with dementia and their carers: a soft serif voice
(Fraunces), handwritten memory captions (Caveat), large readable text, calm
colours, and copy with a soul rather than buzzwords. Every answer shows the
photos it relied on, so nothing is taken on faith.

## Roadmap

- On-device vision + chat (so memories never leave the device)
- People & places ("who was I with?")
- Daily summaries and a caregiver view
- Postgres + pgvector for scale

## Contributing

Contributions are welcome. Please read [`AGENTS.md`](./AGENTS.md) first — it
documents the architecture and the non-negotiable *never fabricate a memory*
rule that any change must preserve.
