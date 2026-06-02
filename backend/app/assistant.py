"""The MindPalace memory assistant: a grounded RAG agent over the person's photos.

Built directly on the Mistral SDK (no extra agent framework) so the tool-calling
loop and the no-hallucination guarantee are easy to audit.

Flow: the model turns the question into `search_memories` tool calls (inferring a
time window for "this morning", a place, content terms), the repository retrieves
matching memories deterministically, and the model writes a warm, grounded reply.
"""

from __future__ import annotations

import json
from datetime import datetime

try:  # SDK layout differs across mistralai versions
    from mistralai import Mistral
except ImportError:  # newer namespace-package layout
    from mistralai.client import Mistral

from . import repository
from .config import settings
from .schemas import ChatMessage, ChatResponse, Memory, SourceMemory

_client: Mistral | None = None

_MAX_TOOL_TURNS = 6

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memories",
        "description": (
            "Search the person's photo-memories. Returns matching memories as JSON. "
            "An EMPTY list means there is no photo matching the filters — in that case "
            "tell the person gently that you have no photo from then; never invent one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "ISO datetime lower bound, e.g. 2026-06-02T05:00:00. "
                    "Use with end_time for ranges like 'morning' or 'yesterday'.",
                },
                "end_time": {
                    "type": "string",
                    "description": "ISO datetime upper bound, e.g. 2026-06-02T12:00:00.",
                },
                "place": {"type": "string", "description": "Part of a place name, e.g. 'park'."},
                "content": {
                    "type": "string",
                    "description": "What is in the photo, e.g. 'dog', 'lunch', 'people'.",
                },
                "limit": {"type": "integer", "description": "Max results (default 10)."},
            },
        },
    },
}

_SYSTEM = """\
You are MindPalace, a gentle memory companion for a person living with memory
loss (such as Alzheimer's or dementia). You help them recall their own day from
photographs they have taken.

How you must behave — these rules are not optional:
- Ground EVERY statement only in memories returned by the search_memories tool.
  Never invent, assume, or embellish a memory. A made-up memory can seriously
  confuse or distress the person, so accuracy matters far more than completeness.
- If the tool returns an empty list, gently say you don't have a photo from that
  time, and perhaps offer to look at another time. Never guess or fill gaps.
- Speak warmly, calmly, and simply: short sentences, one idea at a time, no jargon.
- Be reassuring and never condescending. Call the pictures "your photos".
- When you have them, mention the time of day and the place, to help orient the
  person (e.g. "This morning, around 8 o'clock, you were at the Lakeside Cafe").

Work out time windows from the current date and time given below, then call
search_memories with start_time / end_time. You may call the tool more than once
if a question spans several times or places."""


def _get_client() -> Mistral:
    global _client
    if _client is None:
        _client = Mistral(api_key=settings.mistral_api_key)
    return _client


def _build_system(now: datetime) -> str:
    return (
        _SYSTEM
        + f"\n\nFor reference, right now it is {now:%A, %B %d %Y at %I:%M %p}. "
        "Use this to interpret relative words. As a guide: morning is roughly "
        "05:00-12:00, afternoon 12:00-17:00, evening 17:00-21:00, night 21:00-05:00."
    )


def _memory_for_llm(memory: Memory) -> dict:
    return {
        "id": memory.id,
        "when": memory.captured_at.isoformat() if memory.captured_at else None,
        "place": memory.place_name,
        "caption": memory.caption,
        "tags": memory.tags,
    }


def _friendly_when(memory: Memory) -> str | None:
    if not memory.captured_at:
        return None
    return memory.captured_at.strftime("%a %b %d, %I:%M %p")


def _run_search(args: dict) -> list[Memory]:
    return repository.search_memories(
        start=args.get("start_time"),
        end=args.get("end_time"),
        place=args.get("place"),
        content=args.get("content"),
        limit=int(args.get("limit", 10)),
    )


def chat(message: str, history: list[ChatMessage]) -> ChatResponse:
    client = _get_client()
    now = datetime.now()

    messages: list[dict] = [{"role": "system", "content": _build_system(now)}]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": message})

    looked_at: dict[str, Memory] = {}

    for _ in range(_MAX_TOOL_TURNS):
        response = client.chat.complete(
            model=settings.chat_model,
            max_tokens=1024,
            tools=[_SEARCH_TOOL],
            tool_choice="auto",
            messages=messages,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                if tc.function.name == "search_memories":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    found = _run_search(args)
                    for memory in found:
                        looked_at[memory.id] = memory
                    result = json.dumps([_memory_for_llm(m) for m in found])
                else:
                    result = "[]"
                messages.append(
                    {
                        "role": "tool",
                        "name": tc.function.name,
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            continue

        answer = (msg.content or "").strip()
        sources = [
            SourceMemory(
                id=m.id,
                when=_friendly_when(m),
                where=m.place_name,
                caption=m.caption,
                image_url=m.image_url,
            )
            for m in looked_at.values()
        ]
        return ChatResponse(answer=answer, sources=sources, found_memories=bool(looked_at))

    return ChatResponse(
        answer="I'm sorry, I had a little trouble looking that up just now. Could you ask me again?",
        sources=[],
        found_memories=False,
    )
