"""
Image preprocessing pipeline:
  1. EXIF orientation correction
  2. RGBA → RGB conversion
  3. Image quality validation gate (blur, brightness, resolution)
  4. ImageNet normalization (applied by torchvision transforms during training)
"""
import cv2
import numpy as np
from PIL import Image, ExifTags
from dataclasses import dataclass, field
from typing import Optional
from .config import (
    MIN_RESOLUTION, MAX_INFERENCE_SIZE, BLUR_THRESHOLD, BRIGHTNESS_MIN, BRIGHTNESS_MAX,
)


# ── EXIF orientation ────────────────────────────────────────────────────────

_EXIF_ORIENTATION_TAG = next(
    k for k, v in ExifTags.TAGS.items() if v == "Orientation"
)

_EXIF_ROTATIONS = {3: 180, 6: 270, 8: 90}


def correct_orientation(img: Image.Image) -> tuple[Image.Image, bool]:
    """
    Rotate image to upright orientation based on EXIF metadata.
    Returns (corrected_image, was_rotated).
    """
    try:
        exif = img._getexif()
        if exif:
            orientation = exif.get(_EXIF_ORIENTATION_TAG)
            if orientation in _EXIF_ROTATIONS:
                img = img.rotate(_EXIF_ROTATIONS[orientation], expand=True)
                return img, True
    except (AttributeError, Exception):
        pass
    return img, False


# ── Quality checks ─────────────────────────────────────────────────────────

def blur_score(img_array: np.ndarray) -> float:
    """
    Laplacian variance as a sharpness proxy.
    Higher → sharper. Below BLUR_THRESHOLD the image is considered blurry.
    """
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(img_array: np.ndarray) -> float:
    """Mean pixel value across the image (0-255 scale)."""
    return float(img_array.mean())


@dataclass
class QualityReport:
    passed: bool
    blur: float
    brightness: float
    resolution: tuple[int, int]      # (width, height)
    issues: list[str] = field(default_factory=list)
    fraud_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed":      self.passed,
            "blur_score":  round(self.blur, 2),
            "brightness":  round(self.brightness, 2),
            "resolution":  list(self.resolution),
            "issues":      self.issues,
            "fraud_flags": self.fraud_flags,
        }


def quality_gate(img: Image.Image) -> QualityReport:
    """
    Run all quality checks and return a QualityReport.
    Images that fail the gate should be logged but are still processed —
    the caller decides whether to skip them.
    """
    arr = np.array(img.convert("RGB"))
    w, h = img.size
    issues: list[str] = []
    fraud_flags: list[str] = []

    b_score = blur_score(arr)
    b_mean  = brightness_mean(arr)

    if min(w, h) < MIN_RESOLUTION:
        issues.append(f"resolution_too_small ({w}x{h})")
    if b_score < BLUR_THRESHOLD:
        issues.append(f"blurry (score={b_score:.1f})")
    if b_mean < BRIGHTNESS_MIN:
        issues.append(f"too_dark (mean={b_mean:.1f})")
    if b_mean > BRIGHTNESS_MAX:
        issues.append(f"too_bright (mean={b_mean:.1f})")

    # ── Basic fraud signals ────────────────────────────────────────────────
    # Extremely uniform image — possible screenshot or digital manipulation
    channel_stds = [arr[:, :, c].std() for c in range(3)]
    if max(channel_stds) < 15.0:
        fraud_flags.append("low_pixel_variance (possible solid fill or digitally generated image)")

    return QualityReport(
        passed=len(issues) == 0,
        blur=b_score,
        brightness=b_mean,
        resolution=(w, h),
        issues=issues,
        fraud_flags=fraud_flags,
    )


# ── Full preprocessing pipeline ────────────────────────────────────────────

@dataclass
class PreprocessResult:
    image: Image.Image           # RGB PIL image, ready for model
    quality: QualityReport
    orientation_corrected: bool
    original_size: tuple[int, int]


def preprocess_image(
    image_path: str,
    run_quality_gate: bool = True,
) -> PreprocessResult:
    """
    Load and preprocess a single image:
      - Open and convert to RGB
      - Correct EXIF orientation
      - Run quality gate

    The returned PIL image is suitable for passing to the dataset / model.
    Tensor normalization (ImageNet stats) is handled by torchvision transforms.
    """
    img = Image.open(image_path)
    original_size = img.size

    # RGBA / palette → RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    img, rotated = correct_orientation(img)

    # Cap resolution for CPU inference speed — resize if long side exceeds MAX_INFERENCE_SIZE
    w, h = img.size
    long_side = max(w, h)
    if long_side > MAX_INFERENCE_SIZE:
        scale = MAX_INFERENCE_SIZE / long_side
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    report = quality_gate(img) if run_quality_gate else QualityReport(
        passed=True, blur=0.0, brightness=0.0, resolution=img.size, issues=[]
    )

    return PreprocessResult(
        image=img,
        quality=report,
        orientation_corrected=rotated,
        original_size=original_size,
    )
