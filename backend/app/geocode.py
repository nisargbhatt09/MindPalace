"""Optional reverse geocoding: GPS coordinates -> a friendly place name.

Uses OpenStreetMap Nominatim. Disabled by default (needs internet); failures are
swallowed so ingestion never blocks on it.
"""

from __future__ import annotations

import httpx

from .config import settings

_PLACE_KEYS = (
    "amenity", "shop", "leisure", "tourism", "building",
    "neighbourhood", "suburb", "road", "city", "town", "village",
)


def reverse_geocode(latitude: float | None, longitude: float | None) -> str | None:
    if not settings.enable_geocoding or latitude is None or longitude is None:
        return None
    try:
        response = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": latitude, "lon": longitude, "format": "json", "zoom": 16},
            headers={"User-Agent": "MindPalace/0.1 (local POC)"},
            timeout=5.0,
        )
        response.raise_for_status()
        address = response.json().get("address", {})
        for key in _PLACE_KEYS:
            if key in address:
                return address[key]
    except Exception:
        return None
    return None
