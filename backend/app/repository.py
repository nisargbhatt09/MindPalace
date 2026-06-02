"""Data access + retrieval. All retrieval here is deterministic SQL/Python.

The dominant retrieval signal is structured metadata (when / where); content
matching over the vision-generated caption + tags is the secondary filter. This
mirrors the design insight that "where was I in the morning?" is a time filter,
not an image-similarity query.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from .db import get_connection
from .schemas import Memory


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        filename=row["filename"],
        image_url=f"/api/memories/{row['id']}/image" if row["image_path"] else None,
        created_at=row["created_at"],
        captured_at=row["captured_at"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        place_name=row["place_name"],
        caption=row["caption"],
        tags=json.loads(row["tags"] or "[]"),
        scene=row["scene"],
        activity=row["activity"],
    )


def insert_memory(
    *,
    id: str,
    filename: str,
    image_path: str | None,
    captured_at: str | None,
    latitude: float | None,
    longitude: float | None,
    place_name: str | None,
    caption: str,
    tags: list[str],
    scene: str | None,
    activity: str | None,
) -> Memory:
    created_at = datetime.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, filename, image_path, created_at, captured_at, latitude,
                 longitude, place_name, caption, tags, scene, activity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id, filename, image_path, created_at, captured_at, latitude,
                longitude, place_name, caption, json.dumps(tags or []), scene, activity,
            ),
        )
        conn.commit()
    memory = get_memory(id)
    assert memory is not None
    return memory


def get_memory(memory_id: str) -> Memory | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return _row_to_memory(row) if row else None


def get_image_path(memory_id: str) -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT image_path FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
    return row["image_path"] if row else None


def list_memories(limit: int = 200) -> list[Memory]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY captured_at DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_memory(r) for r in rows]


def search_memories(
    start: str | None = None,
    end: str | None = None,
    place: str | None = None,
    content: str | None = None,
    limit: int = 10,
) -> list[Memory]:
    """Return memories matching every supplied filter.

    Time and place are filtered in SQL. Content is matched over caption + tags
    (any term hits), ranked by number of matching terms. An empty result means
    nothing matched — callers must NOT fabricate a memory in that case.
    """
    clauses: list[str] = []
    params: list[object] = []
    if start:
        clauses.append("captured_at >= ?")
        params.append(start)
    if end:
        clauses.append("captured_at <= ?")
        params.append(end)
    if place:
        clauses.append("LOWER(place_name) LIKE ?")
        params.append(f"%{place.lower()}%")

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM memories{where} ORDER BY captured_at"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    memories = [_row_to_memory(r) for r in rows]

    if content:
        terms = [t for t in content.lower().split() if len(t) > 2]
        if terms:
            def score(memory: Memory) -> int:
                haystack = (memory.caption + " " + " ".join(memory.tags)).lower()
                return sum(1 for term in terms if term in haystack)

            scored = [(score(m), m) for m in memories]
            memories = [m for s, m in sorted(scored, key=lambda x: x[0], reverse=True) if s > 0]

    return memories[:limit]
