"""Application configuration, loaded from the environment (and an optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional convenience; the app runs fine without python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    # Tool-calling chat model for the assistant (grounding depends on good tool use).
    chat_model: str = os.getenv("MINDPALACE_CHAT_MODEL", "mistral-large-latest")
    # Vision model for captioning photos at ingest (Pixtral).
    vision_model: str = os.getenv("MINDPALACE_VISION_MODEL", "pixtral-12b-2409")
    db_path: Path = Path(os.getenv("MINDPALACE_DB", "./mindpalace.db"))
    uploads_dir: Path = Path(os.getenv("MINDPALACE_UPLOADS", "./uploads"))
    enable_geocoding: bool = os.getenv("ENABLE_GEOCODING", "false").lower() == "true"


settings = Settings()
