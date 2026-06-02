"""Ingest pipeline: bytes -> stored image + EXIF + vision description -> DB row.

Combines the deterministic steps (save, EXIF, geocode) with the one LLM step
(vision description) and persists a complete memory.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from . import exif, geocode, repository, vision
from .config import settings
from .schemas import Memory


def ingest_upload(file_bytes: bytes, original_filename: str) -> Memory:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(original_filename).suffix.lower() or ".jpg"
    memory_id = uuid.uuid4().hex
    stored_path = settings.uploads_dir / f"{memory_id}{suffix}"
    stored_path.write_bytes(file_bytes)

    metadata = exif.extract_metadata(stored_path)
    place_name = geocode.reverse_geocode(metadata.latitude, metadata.longitude)
    description = vision.describe_image(stored_path)

    return repository.insert_memory(
        id=memory_id,
        filename=original_filename,
        image_path=str(stored_path),
        captured_at=metadata.captured_at.isoformat(timespec="seconds")
        if metadata.captured_at
        else None,
        latitude=metadata.latitude,
        longitude=metadata.longitude,
        place_name=place_name,
        caption=description.get("caption", ""),
        tags=[str(t).lower() for t in description.get("tags", [])],
        scene=description.get("scene"),
        activity=description.get("activity"),
    )
