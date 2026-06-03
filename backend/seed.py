"""Seed a simulated day of photo-memories so the chat can be demoed without
real photos or any vision API calls.

Run from the backend/ directory:
    python seed.py
"""

from __future__ import annotations

from datetime import date, datetime, time

from app.db import init_db
from app import repository

# A simulated day. Each entry's time-of-day is fixed; the date is anchored to
# *today* at seed time (see main) so demo questions about "today"/"this morning"
# — which the assistant resolves against the current date — actually match.
# No image files (image_path=None); captions + tags + time + place are enough
# to exercise retrieval and chat.
SAMPLE = [
    dict(id="seed_0815", at=time(8, 15), place_name="Lakeside Cafe",
         caption="A cup of coffee and a croissant on a wooden table by a window.",
         tags=["coffee", "croissant", "breakfast", "food", "cafe", "morning"],
         scene="cafe", activity="having breakfast"),
    dict(id="seed_1030", at=time(10, 30), place_name="Riverside Park",
         caption="An elderly man walking a small brown dog along a tree-lined path.",
         tags=["dog", "animal", "pet", "walking", "park", "trees", "outdoors"],
         scene="park", activity="walking a dog"),
    dict(id="seed_1300", at=time(13, 0), place_name="Home",
         caption="A plate of pasta with tomato sauce on the kitchen table.",
         tags=["pasta", "lunch", "food", "meal", "kitchen", "tomato"],
         scene="home kitchen", activity="eating lunch"),
    dict(id="seed_1545", at=time(15, 45), place_name="Greenfield Library",
         caption="Tall bookshelves filled with colorful books in a quiet library.",
         tags=["books", "library", "reading", "bookshelves", "indoors"],
         scene="library", activity="visiting the library"),
    dict(id="seed_1830", at=time(18, 30), place_name="Home",
         caption="A smiling woman and a young child sitting together on a sofa.",
         tags=["people", "family", "child", "woman", "sofa", "smiling", "home"],
         scene="living room", activity="spending time with family"),
]


def main() -> None:
    init_db()
    today = date.today()
    for item in SAMPLE:
        item = dict(item)
        captured_at = datetime.combine(today, item.pop("at")).isoformat(timespec="seconds")
        repository.insert_memory(
            filename=f"{item['id']}.jpg",
            image_path=None,
            captured_at=captured_at,
            latitude=None,
            longitude=None,
            **item,
        )
    print(f"Seeded {len(SAMPLE)} memories for {today.isoformat()}.")


if __name__ == "__main__":
    main()
