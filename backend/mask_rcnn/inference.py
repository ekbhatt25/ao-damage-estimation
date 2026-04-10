"""
Single-image inference pipeline.

Runs both the parts model and the damage model, then cross-references their
masks to determine which parts are damaged and by what damage type.

Output is a structured JSON dict designed to be fed directly into an LLM
for downstream repair estimation, severity assessment, etc.

Usage:
  python -m backend.mask_rcnn.inference --image path/to/car.jpg
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO as UltralyticsYOLO

from .config import (
    PARTS_MODEL_PATH, DAMAGE_MODEL_PATH, YOLO_DAMAGE_MODEL_PATH,
    SEVERITY_MODEL_PATH, SEVERITY_MODEL_HF_REPO, SEVERITY_MODEL_HF_FILE,
    PART_CLASSES, DAMAGE_CLASSES,
    SCORE_THRESHOLD, PART_DAMAGE_OVERLAP_THRESHOLD,
    MODELS_DIR,
)
from .model import build_parts_model, build_damage_model, load_checkpoint
from .preprocess import preprocess_image
import torchvision.transforms.functional as TF


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_yolo_damage_model() -> UltralyticsYOLO:
    if not YOLO_DAMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO damage model not found at {YOLO_DAMAGE_MODEL_PATH}."
        )
    return UltralyticsYOLO(str(YOLO_DAMAGE_MODEL_PATH))


def _run_yolo_damage_model(
    model: UltralyticsYOLO,
    image_path: str,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """Run YOLO damage model; returns detections in the same format as _run_model."""
    results = model(image_path, conf=score_threshold, verbose=False)[0]
    img_h, img_w = results.orig_shape
    class_names = model.names  # {0: 'dent', 1: 'scratch', ...}

    detections = []
    for box in results.boxes:
        score = float(box.conf[0])
        label = int(box.cls[0])
        x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]

        # Synthesise a bbox mask so overlap logic works unchanged
        bin_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        bin_mask[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)] = 1

        detections.append({
            "label":      label,
            "class_name": class_names.get(label, "unknown"),
            "score":      round(score, 4),
            "bbox":       [x1, y1, x2, y2],
            "mask":       bin_mask,
            "mask_area":  int(bin_mask.sum()),
        })
    return detections


def _load_model(mode: str, device: torch.device) -> torch.nn.Module:
    path = PARTS_MODEL_PATH if mode == "parts" else DAMAGE_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train.py --mode {mode} first."
        )
    model = build_parts_model(pretrained=False) if mode == "parts" \
        else build_damage_model(pretrained=False)
    load_checkpoint(model, str(path), device)
    model.to(device)
    model.eval()
    return model


def _run_model(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    class_names: list[str],
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """Run one model on a pre-processed image tensor, return detection dicts."""
    with torch.no_grad():
        preds = model([image_tensor.to(device)])[0]

    scores  = preds["scores"].cpu().numpy()
    labels  = preds["labels"].cpu().numpy()
    masks   = preds["masks"].cpu().numpy()   # (N, 1, H, W) float
    boxes   = preds["boxes"].cpu().numpy()   # (N, 4) x1y1x2y2

    detections = []
    for i in np.where(scores >= score_threshold)[0]:
        bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
        x1, y1, x2, y2 = boxes[i].tolist()
        label = int(labels[i])
        detections.append({
            "label":      label,
            "class_name": class_names[label] if label < len(class_names) else "unknown",
            "score":      round(float(scores[i]), 4),
            "bbox":       [round(x1), round(y1), round(x2), round(y2)],
            "mask":       bin_mask,                  # (H, W) uint8, internal use
            "mask_area":  int(bin_mask.sum()),
        })
    return detections


def _overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Intersection area as a fraction of mask_b's area.
    Answers: "what fraction of the damage falls on this part?"
    """
    intersection = (mask_a & mask_b).sum()
    denom = mask_b.sum()
    return float(intersection) / float(denom + 1e-6)


def _severity_proxy(overlap: float, damage_area: int, part_area: int) -> str:
    """
    Heuristic fallback when the ML severity model is unavailable.
    """
    ratio = damage_area / max(part_area, 1)
    if ratio > 0.40 or overlap > 0.60:
        return "severe"
    if ratio > 0.15 or overlap > 0.25:
        return "moderate"
    return "minor"


# ── ML-based severity classification ──────────────────────────────────────────
# YOLOv8n-cls model from nezahatkorkmaz/car-damage-level-detection-yolov8
# Classes: {0: "01-minor", 1: "02-moderate", 2: "03-severe"}
_SEVERITY_CLASS_MAP = {0: "minor", 1: "moderate", 2: "severe"}


def _load_severity_model() -> UltralyticsYOLO | None:
    """Load the YOLOv8 severity classifier, downloading from HF if needed."""
    if SEVERITY_MODEL_PATH.exists():
        return UltralyticsYOLO(str(SEVERITY_MODEL_PATH))

    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(SEVERITY_MODEL_HF_REPO, SEVERITY_MODEL_HF_FILE)
        import shutil
        shutil.copy2(local, SEVERITY_MODEL_PATH)
        return UltralyticsYOLO(str(SEVERITY_MODEL_PATH))
    except Exception as exc:
        print(f"⚠ Severity model unavailable ({exc}); falling back to heuristic")
        return None


def _classify_severity(
    severity_model: UltralyticsYOLO,
    pil_image: Image.Image,
    damage_bbox: list[int],
    overlap: float,
    damage_area: int,
    part_area: int,
    min_crop_px: int = 32,
) -> str:
    """
    Crop the damage region from *pil_image* and run the severity classifier.
    Falls back to _severity_proxy if the crop is too small or inference fails.
    """
    x1, y1, x2, y2 = damage_bbox
    w, h = x2 - x1, y2 - y1
    if w < min_crop_px or h < min_crop_px:
        return _severity_proxy(overlap, damage_area, part_area)

    try:
        crop = pil_image.crop((x1, y1, x2, y2))
        results = severity_model(crop, verbose=False)
        probs = results[0].probs
        predicted_idx = int(probs.top1)
        return _SEVERITY_CLASS_MAP.get(predicted_idx, "moderate")
    except Exception:
        return _severity_proxy(overlap, damage_area, part_area)


def infer(
    image_path: str,
    parts_model: torch.nn.Module = None,
    yolo_damage_model: UltralyticsYOLO = None,
    device: torch.device = None,
    damage_model: torch.nn.Module = None,  # unused, kept for backwards compat
    severity_model: UltralyticsYOLO = None,
) -> dict:
    """
    Full pipeline for a single image:
      1. Preprocess (orientation, quality gate)
      2. Run parts model
      3. Run damage model
      4. Compute part-damage overlaps  (severity via ML classifier or heuristic)
      5. Return structured JSON-serialisable dict

    models can be passed in for batch use (avoids reloading weights each call).
    """
    if device is None:
        device = get_device()
    if parts_model is None:
        parts_model = _load_model("parts", device)
    if yolo_damage_model is None:
        yolo_damage_model = _load_yolo_damage_model()
    if severity_model is None:
        severity_model = _load_severity_model()
    use_ml_severity = severity_model is not None

    # ── 1. Preprocess ──────────────────────────────────────────────────────
    t_start   = time.perf_counter()
    pre       = preprocess_image(image_path)
    image_t   = TF.to_tensor(pre.image)   # [3, H, W] float32
    preprocess_ms = (time.perf_counter() - t_start) * 1000
    print(f"[TIMING] preprocess:    {preprocess_ms:.0f}ms  size={pre.image.size}")

    # ── 2. Mask R-CNN parts ────────────────────────────────────────────────
    t_parts = time.perf_counter()
    parts   = _run_model(parts_model, image_t, device, PART_CLASSES)
    parts_ms = (time.perf_counter() - t_parts) * 1000
    print(f"[TIMING] mask_rcnn:     {parts_ms:.0f}ms  parts_found={len(parts)}")

    # ── 3. YOLO damage ────────────────────────────────────────────────────
    t_yolo  = time.perf_counter()
    damages = _run_yolo_damage_model(yolo_damage_model, image_path, SCORE_THRESHOLD)
    yolo_ms = (time.perf_counter() - t_yolo) * 1000
    print(f"[TIMING] yolo_damage:   {yolo_ms:.0f}ms  damages_found={len(damages)}")

    infer_ms = parts_ms + yolo_ms

    # ── 4. Cross-reference: which parts are damaged? ───────────────────────
    damaged_part_map: dict[str, dict] = {}  # part_name → summary

    for part in parts:
        for dmg in damages:
            ov = _overlap_ratio(part["mask"], dmg["mask"])
            if ov >= PART_DAMAGE_OVERLAP_THRESHOLD:
                pname = part["class_name"]
                if pname not in damaged_part_map:
                    damaged_part_map[pname] = {
                        "part":             pname,
                        "part_confidence":  part["score"],
                        "part_bbox":        part["bbox"],
                        "part_mask_area_px": part["mask_area"],
                        "damage_types":     [],
                    }
                entry = damaged_part_map[pname]
                if use_ml_severity:
                    severity = _classify_severity(
                        severity_model,
                        pre.image,
                        dmg["bbox"],
                        ov,
                        dmg["mask_area"],
                        part["mask_area"],
                    )
                else:
                    severity = _severity_proxy(ov, dmg["mask_area"], part["mask_area"])
                entry["damage_types"].append({
                    "type":              dmg["class_name"],
                    "confidence":        dmg["score"],
                    "overlap_ratio":     round(ov, 3),
                    "damage_bbox":       dmg["bbox"],
                    "damage_area_px":    dmg["mask_area"],
                    "severity_proxy":    severity,
                })

    # ── 4b. Fallback: if cross-reference yielded nothing but both models fired,
    #         pair each damage with the best-overlapping part (any overlap > 0).
    #         This prevents a silent empty result when spatial overlap is low.
    if not damaged_part_map and parts and damages:
        for dmg in damages:
            best_part = max(parts, key=lambda p: _overlap_ratio(p["mask"], dmg["mask"]))
            ov = _overlap_ratio(best_part["mask"], dmg["mask"])
            pname = best_part["class_name"]
            if pname not in damaged_part_map:
                damaged_part_map[pname] = {
                    "part":              pname,
                    "part_confidence":   best_part["score"],
                    "part_bbox":         best_part["bbox"],
                    "part_mask_area_px": best_part["mask_area"],
                    "damage_types":      [],
                }
            if use_ml_severity:
                severity = _classify_severity(
                    severity_model,
                    pre.image,
                    dmg["bbox"],
                    ov,
                    dmg["mask_area"],
                    best_part["mask_area"],
                )
            else:
                severity = _severity_proxy(ov, dmg["mask_area"], best_part["mask_area"])
            damaged_part_map[pname]["damage_types"].append({
                "type":           dmg["class_name"],
                "confidence":     dmg["score"],
                "overlap_ratio":  round(ov, 3),
                "damage_bbox":    dmg["bbox"],
                "damage_area_px": dmg["mask_area"],
                "severity_proxy": severity,
            })

    total_ms = (time.perf_counter() - t_start) * 1000
    print(f"[TIMING] total_pipeline:{total_ms:.0f}ms  damaged_parts={len(damaged_part_map)}")

    # ── 5. Build output dict ───────────────────────────────────────────────
    w, h = pre.image.size
    output = {
        "image_path": str(image_path),
        "quality_check": pre.quality.to_dict(),
        "preprocessing": {
            "orientation_corrected": pre.orientation_corrected,
            "original_size":         list(pre.original_size),
            "processed_size":        [w, h],
        },
        "parts_detected": [
            {
                "part":           p["class_name"],
                "confidence":     p["score"],
                "bbox":           p["bbox"],
                "mask_area_px":   p["mask_area"],
                "mask_area_ratio": round(p["mask_area"] / max(w * h, 1), 4),
            }
            for p in parts
        ],
        "damages_detected": [
            {
                "damage_type": d["class_name"],
                "confidence":  d["score"],
                "bbox":        d["bbox"],
                "area_px":     d["mask_area"],
            }
            for d in damages
        ],
        "damaged_parts": list(damaged_part_map.values()),
        "summary": {
            "total_parts_detected":   len(parts),
            "total_damages_detected": len(damages),
            "damaged_part_count":     len(damaged_part_map),
            "damaged_part_names":     sorted(damaged_part_map.keys()),
        },
        "timing": {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms":  round(infer_ms, 2),
            "total_ms":      round((time.perf_counter() - t_start) * 1000, 2),
        },
    }

    return output


def infer_and_save(image_path: str, out_path: str = None) -> dict:
    result = infer(image_path)
    if out_path is None:
        stem    = Path(image_path).stem
        out_path = str(MODELS_DIR / f"{stem}_result.json")
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out",   default=None,  help="Path for output JSON")
    args = parser.parse_args()

    result = infer_and_save(args.image, args.out)
    print(json.dumps(result, indent=2, default=str))
