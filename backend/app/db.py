"""SQLite persistence: on-device, zero-config, single file.

A repository pattern (see repository.py) sits on top so this can be swapped for
Postgres + pgvector at production scale without touching the rest of the app.
"""

from __future__ import annotations

import sqlite3

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    image_path  TEXT,
    created_at  TEXT NOT NULL,           -- when ingested (ISO 8601)
    captured_at TEXT,                    -- when the photo was taken, from EXIF (ISO 8601)
    latitude    REAL,
    longitude   REAL,
    place_name  TEXT,
    caption     TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '[]',  -- JSON array of lowercase keywords
    scene       TEXT,
    activity    TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_captured_at ON memories(captured_at);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
