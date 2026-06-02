"""FastAPI app exposing the MindPalace memory assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import ingest, repository
from .assistant import chat as run_chat
from .config import settings
from .db import init_db
from .schemas import ChatRequest, ChatResponse, Memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="MindPalace", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chat_model": settings.chat_model, "vision_model": settings.vision_model}


@app.get("/api/memories", response_model=list[Memory])
def list_memories() -> list[Memory]:
    return repository.list_memories(limit=200)


@app.post("/api/ingest", response_model=Memory)
async def ingest_endpoint(file: UploadFile = File(...)) -> Memory:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    # ingest_upload makes a blocking LLM call; FastAPI runs sync deps in a
    # threadpool, but this endpoint is async, so offload explicitly.
    import anyio

    return await anyio.to_thread.run_sync(
        ingest.ingest_upload, data, file.filename or "upload.jpg"
    )


@app.get("/api/memories/{memory_id}/image")
def get_image(memory_id: str) -> FileResponse:
    path = repository.get_image_path(memory_id)
    if not path:
        raise HTTPException(status_code=404, detail="No image for this memory.")
    return FileResponse(path)


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    return run_chat(request.message, request.history)
