"""Pydantic models for API I/O and internal data passing."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Memory(BaseModel):
    id: str
    filename: str
    image_url: str | None = None
    created_at: datetime
    captured_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    place_name: str | None = None
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    scene: str | None = None
    activity: str | None = None


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class SourceMemory(BaseModel):
    """A memory the assistant looked at, surfaced so the answer is auditable."""

    id: str
    when: str | None = None
    where: str | None = None
    caption: str
    image_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceMemory] = Field(default_factory=list)
    found_memories: bool = False
