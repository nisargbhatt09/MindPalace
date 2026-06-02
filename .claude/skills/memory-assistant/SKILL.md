---
name: memory-assistant
description: >-
  Build, extend, or debug the MindPalace memory assistant — the grounded RAG
  layer that answers a person's natural-language questions ("where was I this
  morning?", "did I see a dog today?") from their photo-memories. Use when adding
  ingest/metadata steps, retrieval filters, agent tools, prompts, or the chat
  loop; or when reviewing any change that touches how answers are produced.
  Enforces the no-fabricated-memories safety rule.
---

# MindPalace memory assistant

This skill captures how to work on the assistant safely. MindPalace helps people
with memory loss (Alzheimer's, dementia) recall their day from their own photos.
Read `AGENTS.md` for full project context first.

## Non-negotiable safety rule: never fabricate a memory

The person relies on this to know what is real. Before shipping ANY change to
prompts, retrieval, tools, or output handling, confirm all of these still hold:

1. Every statement in an answer is grounded in a retrieved memory.
2. Empty retrieval → the assistant says it has no photo from then; it does NOT
   guess, infer, or generalize.
3. The answer exposes which memories it used (auditable by a caregiver).
4. The negative test ("Was I at the beach today?" with no beach photo) yields a
   graceful "I don't have a photo" — not an invented one.

If a change can't preserve these, stop and raise it with the user.

## Mental model

Retrieval is **structured-first**: when/where dominate; content is secondary.

```
question → agent infers filters → search_memories(start, end, place, content)
         → repository filters memories → grounded, validated answer
```

A memory has: `id`, `timestamp`, `place_name`, `latitude/longitude`, `caption`,
`image_path`. The agent answer is a validated object: a warm `answer` string,
a `found_memories` bool, and the list of `memories_used`.

## Common tasks

### Add a memory (ingest)
Caption the image (BLIP), extract EXIF time + GPS, reverse-geocode GPS to a
`place_name`, then store the full memory. For the POC, simulated time+place is
acceptable; real EXIF/geocoding is the production path.

### Add a retrieval filter
Extend the repository's `search` and the agent tool's signature together. Keep
filters composable (all supplied filters must match). Sort time-based results
chronologically. Return an empty list (never a fallback guess) when nothing matches.

### Add a question capability
Prefer teaching the agent to combine existing filters over adding new tools.
Update the system prompt only if the model needs new guidance (e.g. a new
time-of-day convention). Keep the prompt's grounding rules intact.

### Tune the persona
Warm, calm, simple: short sentences, one idea at a time, no jargon, never
condescending. Mention time of day and place to help orient the person. The
persona must never override the grounding rules.

## How to verify a change

Run the demo question set and check each expectation:

| Question | Expected |
|---|---|
| "Where was I in the morning?" | morning photos, with place + time |
| "Did I see any animals today?" | the animal photo(s) only |
| "What did I have for lunch?" | the lunchtime food photo |
| "Was I at the beach today?" (no such photo) | graceful "no photo" — **no fabrication** |
| "What did I do this afternoon?" | afternoon photos summarized |

A change is only good if retrieval is correct AND the negative case stays clean.
