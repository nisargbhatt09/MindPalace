"""Vision captioning at ingest time, via Mistral's Pixtral model.

Replaces a local BLIP model. Produces a plain-language caption PLUS structured,
queryable tags (including broad categories — a dog photo gets `dog, animal, pet`)
so the deterministic keyword retrieval can answer fuzzy questions like
"did I see any animals today?".

Uses JSON mode (rather than tool-forcing) for reliable structured output from
Pixtral. Output is parsed defensively.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

try:  # SDK layout differs across mistralai versions
    from mistralai import Mistral
except ImportError:  # newer namespace-package layout
    from mistralai.client import Mistral

from .config import settings

_client: Mistral | None = None

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_PROMPT = """\
Describe this photo so it can be recalled later as a personal memory.
Respond with ONLY a JSON object with exactly these fields:
- "caption": one or two warm, plain sentences describing the photo.
- "tags": an array of 5-12 lowercase keywords. Include broad categories so general
  queries match (e.g. for a dog include "dog", "animal", "pet").
- "scene": a short setting label, e.g. "park", "restaurant", "home kitchen".
- "activity": a short activity label, e.g. "walking a dog", "eating lunch"."""


def _get_client() -> Mistral:
    global _client
    if _client is None:
        _client = Mistral(api_key=settings.mistral_api_key)
    return _client


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Fall back to extracting the first {...} block.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def describe_image(image_path: Path | str) -> dict:
    """Return {caption, tags, scene?, activity?} for the image."""
    path = Path(image_path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode()
    data_uri = f"data:{media_type};base64,{data}"

    response = _get_client().chat.complete(
        model=settings.vision_model,
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": data_uri},
                ],
            }
        ],
    )

    content = response.choices[0].message.content or ""
    parsed = _parse_json(content if isinstance(content, str) else "")
    return {
        "caption": parsed.get("caption", ""),
        "tags": parsed.get("tags", []) or [],
        "scene": parsed.get("scene"),
        "activity": parsed.get("activity"),
    }
