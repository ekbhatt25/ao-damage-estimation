"""
Fraud signal detection: metadata anomaly detection + duplicate image detection.

check_metadata  — EXIF-based signals (editing software)
check_duplicate — perceptual hash comparison against submissions within the last 10 minutes
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

_HASH_STORE = Path(__file__).parent / "claim_hash_store.json"
_HASH_SIZE = 8
_DUPLICATE_THRESHOLD = 10   # max Hamming distance out of 64 bits
_WINDOW_SECONDS = 60        # 1-minute duplicate window

_EDITING_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity",
    "snapseed", "facetune", "pixelmator", "canva", "paint.net",
]


def _phash(image: Image.Image) -> str:
    """8×8 average perceptual hash → 64-char binary string."""
    grey = image.convert("L").resize((_HASH_SIZE, _HASH_SIZE), Image.LANCZOS)
    pixels = np.array(grey, dtype=float)
    bits = pixels.flatten() > pixels.mean()
    return "".join("1" if b else "0" for b in bits)


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def check_duplicate(image: Image.Image) -> str | None:
    """
    Returns a fraud flag string if the image is visually identical to a
    submission made within the last 10 minutes, else None.
    Always stores the hash + timestamp for future comparisons.
    """
    h = _phash(image)
    now = time.time()

    entries: list[dict] = []
    if _HASH_STORE.exists():
        try:
            raw = json.loads(_HASH_STORE.read_text())
            # Support old format (list of strings) gracefully
            if raw and isinstance(raw[0], str):
                entries = [{"hash": s, "ts": 0} for s in raw]
            else:
                entries = raw
        except Exception:
            entries = []

    # Drop entries older than the window
    entries = [e for e in entries if now - e["ts"] < _WINDOW_SECONDS]

    flag = None
    for entry in entries:
        if _hamming(h, entry["hash"]) <= _DUPLICATE_THRESHOLD:
            flag = "duplicate_image (same photo submitted within the last minute)"
            break

    # Always record this submission
    entries.append({"hash": h, "ts": now})
    try:
        _HASH_STORE.write_text(json.dumps(entries))
    except Exception:
        pass

    return flag


def check_metadata(image_path: str) -> list[str]:
    """
    Returns a list of fraud flag strings based on EXIF anomalies.
    """
    flags = []
    try:
        img = Image.open(image_path)

        try:
            exif = img.getexif()
            exif_dict = dict(exif) if exif else None
        except Exception:
            try:
                exif_dict = img._getexif()
            except Exception:
                exif_dict = None

        if not exif_dict:
            return flags

        # Tag 305 = Software
        software = str(exif_dict.get(305, "")).strip()
        if any(kw in software.lower() for kw in _EDITING_KEYWORDS):
            flags.append(f"editing_software_detected (image processed with {software})")

    except Exception:
        pass

    return flags
