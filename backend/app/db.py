"""SQLite persistence: on-device, zero-config, single file.

A repository pattern (see repository.py) sits on top so this can be swapped for
Postgres + pgvector at production scale without touching the rest of the app.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

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


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection, committing on success and always closing.

    Used as ``with get_connection() as conn:``. A bare ``sqlite3.Connection``
    context manager only manages the transaction (commit/rollback) and leaves
    the connection — and its file handle — open, so this wrapper owns the close.
    """
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
