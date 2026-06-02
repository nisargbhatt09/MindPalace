# MindPalace — Plan

A planning document for evolving MindPalace into an assistive **memory companion
for people living with memory loss** (Alzheimer's, dementia). See `AGENTS.md` for
agent working-guidance and the `memory-assistant` skill for the build workflow.

---

## 1. Problem statement

A person takes photos through their day. The system stores each photo with its
metadata (when + where) and a description, and later answers their natural-language
questions about their own day, conversationally, as a gentle assistant:

> "Where was I in the morning?" · "Did I see a dog today?" · "What did I have for lunch?"

## 2. Key insight (why this isn't image search)

Questions like *"where was I in the morning?"* are driven by **metadata (when /
where)**, not visual similarity:

- "in the morning" is a **time filter** — it lives in EXIF, not in pixels.
- "where" needs a **place**, reverse-geocoded from GPS.

So this is a **metadata-grounded RAG assistant**: retrieval is **structured-first**
(time + place), with semantic caption search as a secondary content filter.
Vector similarity is one tool among several, not the centerpiece.

## 3. The overriding constraint: never fabricate a memory

The person relies on this to know what is real; an invented memory can confuse or
distress them. Therefore:

- Every statement is grounded in a retrieved memory.
- Empty retrieval → "I don't have a photo from then." Never guess or fill gaps.
- Every answer exposes the memories it used, for caregiver verification.
- Accuracy and honesty outrank completeness and fluency.

This is the make-or-break property; the negative test case validates it.

## 4. Architecture

```
INGEST (per photo):
  photo → caption (BLIP) → metadata (EXIF time + GPS → place_name)
        → embedding → store {vector + rich metadata}

ASK (per question):
  question → LLM infers structured filters (time window, place, content)
           → retrieve (structured filter + optional semantic rank)
           → assemble grounded context
           → LLM writes warm, grounded, validated answer
           → multi-turn chat state for follow-ups
```

Query understanding is *implicit*: the model turns fuzzy language into tool-call
arguments (`search_memories(start, end, place, content)`), guided by an injected
"current time" so it can resolve "today", "this morning", "yesterday".

### Per-memory data model

| Field | Source | Used for |
|---|---|---|
| `timestamp` | EXIF `DateTimeOriginal` | temporal queries |
| `latitude`/`longitude` | EXIF GPS | spatial queries |
| `place_name` | reverse-geocode GPS | readable "where" |
| `caption` | BLIP | content + readable context |
| `embedding` | MiniLM/CLIP | semantic content search |
| `image_path` | ingest | optional vision answers |
| `people` *(later)* | face clustering | "who was I with?" |

### Components (target)

| Module | Role |
|---|---|
| `models/metadata_extractor.py` | EXIF time + GPS |
| `models/geocoder.py` | GPS → place name |
| `models/caption_model.py` *(exists)* | image → caption |
| `assistant/repository.py` | structured + semantic retrieval (agent dependency) |
| `assistant/agent.py` | LLM + tools + care persona + validated output |
| `assistant/schemas.py` | Pydantic models (Memory, AssistantResponse) |
| `database/vector_store.py` *(exists)* | persistence (Pinecone or local) |

## 5. POC

### Goal
Prove that fuzzy natural-language questions over ~15 photo-memories return
correct, grounded answers — **including correctly saying "no photo" when there is
none.** If that works, the concept is validated; the rest is engineering.

### In scope
Simulated one-day dataset (time + place + caption), local storage (JSON /
in-memory), structured + semantic retrieval, one LLM with the care persona, a CLI
chat loop, a fixed demo question set including a negative case.

### Out of scope (defer)
Mobile app, camera capture, real-time sync, Pinecone/cloud, auth, encryption,
face recognition, caregiver dashboard, multi-day scale.

### Data shortcut
Real phone EXIF/GPS is fiddly. For the POC, **simulate one realistic day**: ~15
images each tagged with `timestamp`, `place_name`, and a BLIP caption, spanning
morning/afternoon/evening across a few places. Real EXIF + geocoding becomes
Phase 1 of the real build.

### Build steps
1. **Seed data** — caption ~15 images with BLIP, attach time + place, write `memories.json`.
2. **Retrieval** — `search_memories(start, end, place, content)` over the data.
3. **Agent** — one LLM, the `search_memories` tool, anti-hallucination + care prompt, structured output.
4. **CLI chat** — ask a question, print the grounded answer + cited memories.

### Demo question set

| Question | Tests | Expected |
|---|---|---|
| "Where was I in the morning?" | temporal + place | morning photos w/ place + time |
| "Did I see any animals today?" | semantic content | the animal photo(s) |
| "What did I have for lunch?" | temporal + content | the lunchtime food |
| "Was I at the beach today?" | **no match** | graceful "no photo" — **no fabrication** |
| "What did I do this afternoon?" | window summary | afternoon photos summarized |

### Success criteria
- Correct photos retrieved for time/place/content questions.
- Answers are warm, simple, and mention when/where.
- **Zero fabricated memories** (the negative case is decisive).
- Every answer lists the memories it relied on.

### Sizing & risks
- **Effort:** ~1–2 days (BLIP captioning already exists; mostly retrieval + agent + CLI).
- **Top risk:** the model inventing/over-claiming. Mitigation: strict grounding
  prompt + structured output forcing source listing + the negative test in the demo.
- **Secondary risk:** caption quality (BLIP misses detail). Acceptable for a POC; note as a limitation.

## 6. Roadmap beyond the POC

1. **Real ingest** — EXIF time + GPS + reverse-geocoding; real photos.
2. **Persistence & scale** — chosen vector store; multi-day, metadata-filtered retrieval.
3. **Conversational assistant** — multi-turn chat, the care persona, daily summaries.
4. **People & places** — face clustering ("who was I with?"), frequent-place naming.
5. **Caregiver mode** — timelines, summaries, oversight.
6. **Hardening** — privacy/encryption, accessibility, on-device options.

## 7. Open decisions (confirm with the user before assuming)

- **Generation:** vision LLM (sees images) vs caption+metadata grounded (text) vs hybrid.
- **Storage:** Pinecone (cloud) vs local store (privacy for health-adjacent data).
- **LLM provider:** Claude vs OpenAI vs local/open model.
- **Agent glue:** raw SDK + Pydantic vs PydanticAI vs provider-agnostic abstraction
  (architecture identical across these — only glue code differs).
- **POC metadata:** simulated (fastest) vs real EXIF from actual photos (more convincing).
