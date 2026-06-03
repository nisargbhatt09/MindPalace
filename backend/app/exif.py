"""Deterministic extraction of capture time + GPS from image EXIF data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


@dataclass
class ImageMetadata:
    captured_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None


def _to_decimal_degrees(dms) -> float:
    degrees, minutes, seconds = dms
    return float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0


def extract_metadata(image_path: Path | str) -> ImageMetadata:
    """Read EXIF capture time and GPS. Missing fields come back as None."""
    metadata = ImageMetadata()
    try:
        image = Image.open(image_path)
        exif = image._getexif() or {}
    except Exception:
        return metadata

    readable = {TAGS.get(tag, tag): value for tag, value in exif.items()}

    raw_time = readable.get("DateTimeOriginal") or readable.get("DateTime")
    if raw_time:
        try:
            metadata.captured_at = datetime.strptime(raw_time, "%Y:%m:%d %H:%M:%S")
        except (ValueError, TypeError):
            pass

    gps = readable.get("GPSInfo")
    if gps:
        # Some cameras store GPSInfo as an IFD offset (an int) rather than a
        # mapping, so guard the whole parse — never let a malformed tag abort
        # ingestion; missing coordinates simply come back as None.
        try:
            gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps.items()}
            if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                lat = _to_decimal_degrees(gps_data["GPSLatitude"])
                if gps_data.get("GPSLatitudeRef") == "S":
                    lat = -lat
                lng = _to_decimal_degrees(gps_data["GPSLongitude"])
                if gps_data.get("GPSLongitudeRef") == "W":
                    lng = -lng
                metadata.latitude, metadata.longitude = lat, lng
        except (AttributeError, ValueError, TypeError, ZeroDivisionError):
            pass

    return metadata
