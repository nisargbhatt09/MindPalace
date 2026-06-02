"""Seed a simulated day of photo-memories so the chat can be demoed without
real photos or any vision API calls.

Run from the backend/ directory:
    python seed.py
"""

from __future__ import annotations

from app.db import init_db
from app import repository

# A simulated day, captured on 2026-06-02. No image files (image_path=None);
# captions + tags + time + place are enough to exercise retrieval and chat.
SAMPLE = [
    dict(id="seed_0815", captured_at="2026-06-02T08:15:00", place_name="Lakeside Cafe",
         caption="A cup of coffee and a croissant on a wooden table by a window.",
         tags=["coffee", "croissant", "breakfast", "food", "cafe", "morning"],
         scene="cafe", activity="having breakfast"),
    dict(id="seed_1030", captured_at="2026-06-02T10:30:00", place_name="Riverside Park",
         caption="An elderly man walking a small brown dog along a tree-lined path.",
         tags=["dog", "animal", "pet", "walking", "park", "trees", "outdoors"],
         scene="park", activity="walking a dog"),
    dict(id="seed_1300", captured_at="2026-06-02T13:00:00", place_name="Home",
         caption="A plate of pasta with tomato sauce on the kitchen table.",
         tags=["pasta", "lunch", "food", "meal", "kitchen", "tomato"],
         scene="home kitchen", activity="eating lunch"),
    dict(id="seed_1545", captured_at="2026-06-02T15:45:00", place_name="Greenfield Library",
         caption="Tall bookshelves filled with colorful books in a quiet library.",
         tags=["books", "library", "reading", "bookshelves", "indoors"],
         scene="library", activity="visiting the library"),
    dict(id="seed_1830", captured_at="2026-06-02T18:30:00", place_name="Home",
         caption="A smiling woman and a young child sitting together on a sofa.",
         tags=["people", "family", "child", "woman", "sofa", "smiling", "home"],
         scene="living room", activity="spending time with family"),
]


def main() -> None:
    init_db()
    for item in SAMPLE:
        repository.insert_memory(
            filename=f"{item['id']}.jpg",
            image_path=None,
            latitude=None,
            longitude=None,
            **item,
        )
    print(f"Seeded {len(SAMPLE)} memories.")


if __name__ == "__main__":
    main()
