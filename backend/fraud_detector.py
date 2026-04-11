"""
Fraud signal detection: metadata anomaly detection + duplicate image detection.

check_metadata  — EXIF-based signals (no camera data, editing software)
check_duplicate — perceptual hash comparison against prior submissions
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

_HASH_STORE = Path(__file__).parent / "claim_hash_store.json"
_HASH_SIZE = 8
_DUPLICATE_THRESHOLD = 10  # max Hamming distance out of 64 bits

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
    previously submitted image, else None. Always stores the hash for
    future comparisons.
    """
    h = _phash(image)

    hashes: list[str] = []
    if _HASH_STORE.exists():
        try:
            hashes = json.loads(_HASH_STORE.read_text())
        except Exception:
            hashes = []

    flag = None
    for stored in hashes:
        if _hamming(h, stored) <= _DUPLICATE_THRESHOLD:
            flag = "duplicate_image (visually identical to a previously submitted photo)"
            break

    if h not in hashes:
        hashes.append(h)
        try:
            _HASH_STORE.write_text(json.dumps(hashes))
        except Exception:
            pass

    return flag


def check_metadata(image_path: str) -> list[str]:
    """
    Returns a list of fraud flag strings based on EXIF anomalies.
    Screenshots and downloaded images have no EXIF.
    Edited images often carry an editing software tag.
    """
    flags = []
    try:
        img = Image.open(image_path)

        # Try modern Pillow API first, fall back to legacy
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

        # Tags 271 = Make, 272 = Model
        make = str(exif_dict.get(271, "")).strip()
        model = str(exif_dict.get(272, "")).strip()
        if not make and not model:
            flags.append(
                "no_camera_info (no device make/model — "
                "common in edited or AI-generated images)"
            )

    except Exception:
        pass

    return flags
