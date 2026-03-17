"""
Evaluation module:
  - COCO-style mAP (IoU 0.50:0.95) via pycocotools
  - Per-class precision / recall at IoU 0.50
  - Inference latency benchmark (mean / std / p95)

Usage:
  python -m backend.mask_rcnn.evaluate --mode parts
  python -m backend.mask_rcnn.evaluate --mode damage
"""
import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils

from .config import (
    PARTS_MODEL_PATH, DAMAGE_MODEL_PATH, CARDD_MODEL_PATH,
    PART_CLASSES, DAMAGE_CLASSES, CARDD_CLASSES,
    CARDD_VAL_ANN, CARDD_VAL_DIR,
    SCORE_THRESHOLD, NMS_IOU_THRESHOLD,
    LATENCY_WARMUP_RUNS, LATENCY_BENCH_RUNS,
    MODELS_DIR,
)
from .dataset import CarDataset
from .dataset_coco import CocoCarDataset
from .model import build_parts_model, build_damage_model, build_cardd_model, load_checkpoint


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── COCO format conversion ─────────────────────────────────────────────────

def dataset_to_coco_gt(dataset: CarDataset) -> dict:
    """Convert our dataset split to a COCO-format ground-truth dict."""
    class_names = PART_CLASSES if dataset.mode == "parts" else DAMAGE_CLASSES

    categories = [
        {"id": i, "name": name}
        for i, name in enumerate(class_names)
        if i > 0  # skip background
    ]
    images, annotations = [], []
    ann_id = 1

    for idx in range(len(dataset)):
        img_t, target = dataset[idx]
        _, h, w = img_t.shape

        images.append({"id": idx, "width": w, "height": h})

        masks  = target["masks"].numpy()   # (N, H, W)
        boxes  = target["boxes"].numpy()   # (N, 4) x1y1x2y2
        labels = target["labels"].numpy()  # (N,)

        for i in range(len(labels)):
            label = int(labels[i])
            if label == 0:
                continue

            # Encode mask with pycocotools RLE for efficiency
            mask_fortran = np.asfortranarray(masks[i])
            rle = coco_mask_utils.encode(mask_fortran)
            rle["counts"] = rle["counts"].decode("utf-8")

            x1, y1, x2, y2 = boxes[i]
            annotations.append({
                "id":          ann_id,
                "image_id":    idx,
                "category_id": label,
                "segmentation": rle,
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area":        float((x2 - x1) * (y2 - y1)),
                "iscrowd":     0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def run_inference(
    model: torch.nn.Module,
    dataset: CarDataset,
    device: torch.device,
) -> list[dict]:
    """Run model over the dataset, return COCO-format detection results."""
    model.eval()
    results = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            img_t, _ = dataset[idx]
            images = [img_t.to(device)]

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                preds = model(images)[0]

            scores  = preds["scores"].cpu().numpy()
            labels  = preds["labels"].cpu().numpy()
            masks   = preds["masks"].cpu().numpy()  # (N, 1, H, W) float
            boxes   = preds["boxes"].cpu().numpy()

            keep = scores >= SCORE_THRESHOLD
            for i in np.where(keep)[0]:
                bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                mask_f   = np.asfortranarray(bin_mask)
                rle      = coco_mask_utils.encode(mask_f)
                rle["counts"] = rle["counts"].decode("utf-8")

                x1, y1, x2, y2 = boxes[i]
                results.append({
                    "image_id":    idx,
                    "category_id": int(labels[i]),
                    "segmentation": rle,
                    "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score":       float(scores[i]),
                })

    return results


# ── mAP via pycocotools ────────────────────────────────────────────────────

def compute_map(gt_dict: dict, dt_list: list[dict]) -> dict:
    """
    Run COCOeval and return summary stats.
    Returns dict with 'mAP', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large'.
    """
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if not dt_list:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    coco_dt = coco_gt.loadRes(dt_list)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats  # 12-element array, see COCOeval docs
    return {
        "mAP":       float(stats[0]),
        "mAP_50":    float(stats[1]),
        "mAP_75":    float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_med":   float(stats[4]),
        "mAP_large": float(stats[5]),
    }


# ── Per-class precision / recall at IoU 0.50 ──────────────────────────────

def compute_per_class_metrics(
    gt_dict: dict,
    dt_list: list[dict],
    class_names: list[str],
) -> list[dict]:
    """
    For each class, compute precision and recall at IoU threshold 0.50.
    Returns list of {"class", "precision", "recall", "f1", "n_gt", "n_pred"}.
    """
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if not dt_list:
        return []

    coco_dt = coco_gt.loadRes(dt_list)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.params.iouThrs = np.array([0.50])
    evaluator.evaluate()
    evaluator.accumulate()

    # evaluator.eval['precision'] shape: [T, R, K, A, M]
    # T=iouThrs, R=recall thresholds, K=categories, A=area, M=maxDets
    precision = evaluator.eval["precision"]  # [1, 101, K, 4, 3]
    recall_arr = evaluator.eval["recall"]    # [1, K, 4, 3]

    cat_ids    = evaluator.params.catIds
    rows = []
    for k_idx, cat_id in enumerate(cat_ids):
        name = class_names[cat_id] if cat_id < len(class_names) else str(cat_id)

        # precision: mean over recall thresholds (ignoring -1 sentinel)
        p_vals = precision[0, :, k_idx, 0, 2]
        p_vals = p_vals[p_vals > -1]
        prec   = float(p_vals.mean()) if len(p_vals) else 0.0

        # recall: max recall achieved
        r_val = recall_arr[0, k_idx, 0, 2]
        rec   = float(r_val) if r_val > -1 else 0.0

        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        n_gt   = sum(1 for a in gt_dict["annotations"] if a["category_id"] == cat_id)
        n_pred = sum(1 for d in dt_list if d["category_id"] == cat_id)

        rows.append({
            "class":     name,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "n_gt":      n_gt,
            "n_pred":    n_pred,
        })

    return sorted(rows, key=lambda r: r["class"])


# ── Latency benchmark ──────────────────────────────────────────────────────

def benchmark_latency(
    model: torch.nn.Module,
    dataset: CarDataset,
    device: torch.device,
) -> dict:
    """
    Measures end-to-end inference latency (preprocessing already done in dataset).
    Reports mean, std, and p95 in milliseconds.
    """
    model.eval()
    # Pre-fetch a batch of images to avoid disk I/O noise
    samples = [dataset[i][0].to(device) for i in range(min(10, len(dataset)))]

    # Warm up
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP_RUNS):
            _ = model([samples[0]])
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    with torch.no_grad():
        for i in range(LATENCY_BENCH_RUNS):
            img = samples[i % len(samples)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model([img])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    return {
        "mean_ms":  round(float(arr.mean()), 2),
        "std_ms":   round(float(arr.std()), 2),
        "p50_ms":   round(float(np.percentile(arr, 50)), 2),
        "p95_ms":   round(float(np.percentile(arr, 95)), 2),
        "p99_ms":   round(float(np.percentile(arr, 99)), 2),
        "n_runs":   LATENCY_BENCH_RUNS,
    }


# ── Full evaluation pipeline ───────────────────────────────────────────────

def evaluate_cardd() -> dict:
    """
    Evaluate the CarDD model using the official val2017 split.
    Uses the COCO JSON annotations directly as ground truth.
    """
    device     = get_device()
    model_path = CARDD_MODEL_PATH

    print(f"\n{'='*60}")
    print(f"Evaluating mode : cardd")
    print(f"Model path      : {model_path}")
    print(f"Device          : {device}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py --mode cardd first."
        )

    model = build_cardd_model(pretrained=False)
    load_checkpoint(model, str(model_path), device)
    model.to(device)
    model.eval()

    val_ds = CocoCarDataset("val", augment=False)
    print(f"Val images      : {len(val_ds)}")

    # ── Run inference, produce COCO-format results ─────────────────────
    print("Running inference on val set...")
    results_dt = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(val_ds)):
            img_t, target = val_ds[idx]
            img_id = int(target["image_id"].item())

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                preds = model([img_t.to(device)])[0]

            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            masks  = preds["masks"].cpu().numpy()   # (N, 1, H, W)
            boxes  = preds["boxes"].cpu().numpy()

            for i in np.where(scores >= SCORE_THRESHOLD)[0]:
                bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                rle = coco_mask_utils.encode(np.asfortranarray(bin_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                x1, y1, x2, y2 = boxes[i]
                results_dt.append({
                    "image_id":    img_id,
                    "category_id": int(labels[i]),
                    "segmentation": rle,
                    "bbox":        [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score":       float(scores[i]),
                })

    print(f"  Detections: {len(results_dt)}")

    # ── mAP using official val annotations ────────────────────────────
    print("Computing mAP...")
    coco_gt = val_ds.get_coco()
    map_results = {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
                   "mAP_small": 0.0, "mAP_med": 0.0, "mAP_large": 0.0}
    if results_dt:
        coco_dt = coco_gt.loadRes(results_dt)
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats
        map_results = {
            "mAP":       float(s[0]),
            "mAP_50":    float(s[1]),
            "mAP_75":    float(s[2]),
            "mAP_small": float(s[3]),
            "mAP_med":   float(s[4]),
            "mAP_large": float(s[5]),
        }

    # ── Per-class ──────────────────────────────────────────────────────
    print("Computing per-class metrics...")
    gt_dict = {
        "images":      val_ds.coco.loadImgs(val_ds.coco.getImgIds()),
        "annotations": val_ds.coco.loadAnns(val_ds.coco.getAnnIds()),
        "categories":  val_ds.coco.loadCats(val_ds.coco.getCatIds()),
    }
    per_class = compute_per_class_metrics(gt_dict, results_dt, CARDD_CLASSES)

    # ── Latency ────────────────────────────────────────────────────────
    print("Benchmarking latency...")
    latency = benchmark_latency(model, val_ds, device)

    results = {
        "mode":      "cardd",
        "map":       map_results,
        "per_class": per_class,
        "latency":   latency,
    }

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"mAP (0.50:0.95) : {map_results['mAP']:.4f}")
    print(f"mAP@0.50        : {map_results['mAP_50']:.4f}")
    print(f"mAP@0.75        : {map_results['mAP_75']:.4f}")
    print(f"\nPer-class (IoU=0.50):")
    print(f"  {'Class':<16} {'Prec':>6} {'Rec':>6} {'F1':>6} {'GT':>5} {'Pred':>5}")
    print(f"  {'-'*48}")
    for r in per_class:
        print(f"  {r['class']:<16} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['n_gt']:>5} {r['n_pred']:>5}")
    print(f"\nLatency ({latency['n_runs']} runs):")
    print(f"  Mean: {latency['mean_ms']} ms  Std: {latency['std_ms']} ms  p95: {latency['p95_ms']} ms")

    out_path = MODELS_DIR / "cardd_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    return results


def evaluate(mode: Literal["parts", "damage", "cardd"]) -> dict:
    if mode == "cardd":
        return evaluate_cardd()
    device     = get_device()
    model_path = PARTS_MODEL_PATH if mode == "parts" else DAMAGE_MODEL_PATH
    class_names = PART_CLASSES if mode == "parts" else DAMAGE_CLASSES

    print(f"\n{'='*60}")
    print(f"Evaluating mode : {mode}")
    print(f"Model path      : {model_path}")
    print(f"Device          : {device}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first."
        )

    model = build_parts_model(pretrained=False) if mode == "parts" \
        else build_damage_model(pretrained=False)
    load_checkpoint(model, str(model_path), device)
    model.to(device)
    model.eval()

    val_ds  = CarDataset(mode, "val", augment=False)
    print(f"Val images      : {len(val_ds)}")

    # Build GT in COCO format
    print("\nBuilding ground-truth COCO dict...")
    gt_dict = dataset_to_coco_gt(val_ds)
    print(f"  GT annotations: {len(gt_dict['annotations'])}")

    # Run inference
    print("Running inference on val set...")
    dt_list = run_inference(model, val_ds, device)
    print(f"  Detections:     {len(dt_list)}")

    # mAP
    print("\nComputing mAP...")
    map_results = compute_map(gt_dict, dt_list)

    # Per-class metrics
    print("Computing per-class metrics...")
    per_class = compute_per_class_metrics(gt_dict, dt_list, class_names)

    # Latency
    print("Benchmarking latency...")
    latency = benchmark_latency(model, val_ds, device)

    results = {
        "mode":      mode,
        "map":       map_results,
        "per_class": per_class,
        "latency":   latency,
    }

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"mAP (0.50:0.95) : {map_results['mAP']:.4f}  "
          f"{'✓ ≥0.80' if map_results['mAP'] >= 0.80 else '✗ <0.80 (target)'}")
    print(f"mAP@0.50        : {map_results['mAP_50']:.4f}")
    print(f"mAP@0.75        : {map_results['mAP_75']:.4f}")
    print(f"\nPer-class (IoU=0.50):")
    print(f"  {'Class':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'GT':>5} {'Pred':>5}")
    print(f"  {'-'*50}")
    for r in per_class:
        print(f"  {r['class']:<18} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['n_gt']:>5} {r['n_pred']:>5}")

    print(f"\nLatency ({latency['n_runs']} runs):")
    print(f"  Mean: {latency['mean_ms']} ms  Std: {latency['std_ms']} ms  "
          f"p95: {latency['p95_ms']} ms")

    # ── Performance gap notes ───────────────────────────────────────────
    target_map = 0.80
    if map_results["mAP"] < target_map:
        gap = target_map - map_results["mAP"]
        print(f"\n⚠  Performance gap: {gap:.4f} below target mAP of {target_map}")
        print("   Likely causes:")
        print("   • Dataset size (~800–1000 images) is small for 8–21 classes")
        print("   • High visual similarity between damage types (Dent vs Cracked)")
        print("   • High class imbalance (some parts/damages rarely annotated)")
        print("   • Some images are blurry or low-resolution")
        print("   Suggested mitigations:")
        print("   • Collect 3–5× more annotated images")
        print("   • Use pseudo-labelling on unlabelled images")
        print("   • Apply more aggressive augmentation (Mosaic, CutMix)")
        print("   • Try Swin-T backbone (better small-object detection)")

    # Save results
    out_path = MODELS_DIR / f"{mode}_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parts", "damage", "cardd"], required=True)
    args = parser.parse_args()
    evaluate(args.mode)
