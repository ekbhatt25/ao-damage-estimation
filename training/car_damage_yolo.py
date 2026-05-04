"""
CAR DAMAGE DETECTION - YOLOv8 Training & Evaluation Script
===========================================================
Expects the following folder structure relative to the parent of backend/:

    parent_folder/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── backend/
        └── car_damage_yolo.py   <-- this script

Outputs results to: parent_folder/yolo_damage_results.txt
Saves best model to: parent_folder/best_car_damage_yolo.pt
"""

import os
import time
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # backend/
PROJECT_ROOT = SCRIPT_DIR.parent                      # parent folder

TRAIN_IMG_DIR = PROJECT_ROOT / "train" / "images"
TRAIN_LBL_DIR = PROJECT_ROOT / "train" / "labels"
TEST_IMG_DIR = PROJECT_ROOT / "test" / "images"
TEST_LBL_DIR = PROJECT_ROOT / "test" / "labels"

RESULTS_FILE = PROJECT_ROOT / "yolo_damage_results.txt"
MODEL_SAVE_PATH = PROJECT_ROOT / "best_car_damage_yolo.pt"

CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
NUM_CLASSES = len(CLASS_NAMES)

# Training hyperparameters
EPOCHS = 20
IMG_SIZE = 416
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


# ──────────────────────────────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────────────────────────────
def correct_angle(image_np: np.ndarray) -> np.ndarray:
    """
    Detect and correct rotation using Hough line detection.
    If the dominant angle is small (< 15°), deskew the image.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None:
        return image_np

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)

    # Only correct if the skew is small enough to be accidental
    if abs(median_angle) > 15 or abs(median_angle) < 0.5:
        return image_np

    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(image_np, rotation_matrix, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return corrected


def validate_image(image_path: str, min_size: int = 50) -> bool:
    """Check that an image file is valid and meets minimum quality."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        if h < min_size or w < min_size:
            return False
        return True
    except Exception:
        return False


def preprocess_image(image_path: str) -> np.ndarray | None:
    """
    Load an image, validate it, and apply angle correction.
    Returns the corrected BGR image or None if invalid.
    """
    if not validate_image(image_path):
        return None
    img = cv2.imread(image_path)
    img = correct_angle(img)
    return img


def preprocess_dataset(img_dir: Path, output_dir: Path) -> int:
    """
    Apply preprocessing to every image in img_dir and write results to
    output_dir.  Returns the number of images successfully preprocessed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for img_file in sorted(img_dir.iterdir()):
        if img_file.suffix.lower() not in image_extensions:
            continue
        processed = preprocess_image(str(img_file))
        if processed is None:
            skipped += 1
            continue
        cv2.imwrite(str(output_dir / img_file.name), processed)
        count += 1

    print(f"  Preprocessed {count} images, skipped {skipped} invalid images.")
    return count


# ──────────────────────────────────────────────────────────────────────
# Dataset YAML Creation
# ──────────────────────────────────────────────────────────────────────
def create_dataset_yaml(train_images: Path, test_images: Path) -> str:
    """Write the YOLO dataset YAML and return its path."""
    yaml_path = str(PROJECT_ROOT / "car_damage.yaml")
    data_yaml = {
        "train": str(train_images),
        "val": str(test_images),       # use test as val for this setup
        "test": str(test_images),
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    print(f"Dataset YAML written to {yaml_path}")
    return yaml_path


# ──────────────────────────────────────────────────────────────────────
# Ground Truth Parsing (for evaluation metrics matching DETR output)
# ──────────────────────────────────────────────────────────────────────
def parse_yolo_label(label_path: str, img_w: int, img_h: int):
    """
    Parse a YOLO label file and return list of (class_id, x_min, y_min, x_max, y_max).
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x_min = (cx - bw / 2) * img_w
            y_min = (cy - bh / 2) * img_h
            x_max = (cx + bw / 2) * img_w
            y_max = (cy + bh / 2) * img_h
            boxes.append((cls_id, x_min, y_min, x_max, y_max))
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes in (x_min, y_min, x_max, y_max) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Match predicted boxes to ground truth using greedy IoU matching.
    Returns lists of (true_class_name, pred_class_name) pairs.
    gt_boxes / pred_boxes: list of (class_id, x_min, y_min, x_max, y_max)
    """
    all_true = []
    all_pred = []
    matched_gt = set()
    matched_pred = set()

    # Build IoU matrix
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = compute_iou(pb[1:], gb[1:])
            if iou >= iou_thresh:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou_val, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        all_true.append(CLASS_NAMES[gt_boxes[gi][0]])
        all_pred.append(CLASS_NAMES[pred_boxes[pi][0]])

    # Unmatched ground truths → false negatives
    for gi, gb in enumerate(gt_boxes):
        if gi not in matched_gt:
            all_true.append(CLASS_NAMES[gb[0]])
            all_pred.append("background")

    # Unmatched predictions → false positives
    for pi, pb in enumerate(pred_boxes):
        if pi not in matched_pred:
            all_true.append("background")
            all_pred.append(CLASS_NAMES[pb[0]])

    return all_true, all_pred


# ──────────────────────────────────────────────────────────────────────
# Detailed Evaluation (DETR-style output + YOLO mAP + latency)
# ──────────────────────────────────────────────────────────────────────
def evaluate_and_write_results(
    model,
    yaml_path: str,
    test_img_dir: Path,
    test_lbl_dir: Path,
    results_path: Path,
    training_time: float,
):
    """Run evaluation and write a comprehensive results file."""

    print("\n--- Running validation metrics ---")
    val_metrics = model.val(data=yaml_path, split="test", verbose=False)

    # ── YOLO native metrics ──
    map50_95 = val_metrics.box.map
    map50 = val_metrics.box.map50
    map75 = val_metrics.box.map75

    yolo_precision = getattr(val_metrics.box, "p", [0])
    yolo_recall = getattr(val_metrics.box, "r", [0])
    yolo_f1 = getattr(val_metrics.box, "f1", [0])

    mean_precision = float(np.mean(yolo_precision)) if hasattr(yolo_precision, "__iter__") else float(yolo_precision)
    mean_recall = float(np.mean(yolo_recall)) if hasattr(yolo_recall, "__iter__") else float(yolo_recall)
    mean_f1 = float(np.mean(yolo_f1)) if hasattr(yolo_f1, "__iter__") else float(yolo_f1)

    # ── IoU-matched per-image evaluation (DETR-style) ──
    print("--- Running per-image IoU-matched evaluation ---")
    all_true_names = []
    all_pred_names = []
    latencies = []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    test_images = sorted([
        f for f in test_img_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    for img_path in test_images:
        # Ground truth
        label_file = test_lbl_dir / (img_path.stem + ".txt")
        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size
        gt_boxes = parse_yolo_label(str(label_file), img_w, img_h)

        # Prediction with latency timing
        t_start = time.perf_counter()
        results = model.predict(str(img_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
        t_end = time.perf_counter()
        latencies.append(t_end - t_start)

        pred_boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append((cls_id, x1, y1, x2, y2))

        true_batch, pred_batch = match_predictions_to_gt(gt_boxes, pred_boxes, IOU_THRESHOLD)
        all_true_names.extend(true_batch)
        all_pred_names.extend(pred_batch)

    # ── Latency statistics ──
    latencies_np = np.array(latencies)
    latency_mean = float(np.mean(latencies_np)) * 1000   # ms
    latency_std = float(np.std(latencies_np)) * 1000
    latency_median = float(np.median(latencies_np)) * 1000
    latency_p95 = float(np.percentile(latencies_np, 95)) * 1000
    latency_min = float(np.min(latencies_np)) * 1000
    latency_max = float(np.max(latencies_np)) * 1000

    # ── Build the results text ──
    all_class_names = sorted(set(all_true_names + all_pred_names) - {"background"})

    filtered_true = []
    filtered_pred = []
    for t, p in zip(all_true_names, all_pred_names):
        if t != "background" or p != "background":
            filtered_true.append(t)
            filtered_pred.append(p)

    lines = []
    lines.append("=" * 70)
    lines.append("CAR DAMAGE DETECTION - EVALUATION RESULTS (YOLOv8)")
    lines.append("=" * 70)
    lines.append("")

    # Training config
    lines.append("TRAINING CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Model:            YOLOv8m (yolov8m.pt)")
    lines.append(f"Epochs:           {EPOCHS}")
    lines.append(f"Image Size:       {IMG_SIZE}")
    lines.append(f"Conf Threshold:   {CONFIDENCE_THRESHOLD}")
    lines.append(f"IoU Threshold:    {IOU_THRESHOLD}")
    lines.append(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    lines.append(f"Training Time:    {training_time:.1f} seconds ({training_time/60:.1f} min)")
    lines.append(f"Test Images:      {len(test_images)}")
    lines.append("")
    lines.append("")

    # YOLO native mAP metrics
    lines.append("YOLO mAP METRICS")
    lines.append("-" * 40)
    lines.append(f"mAP@0.5:0.95:  {map50_95:.4f}")
    lines.append(f"mAP@0.5:       {map50:.4f}")
    lines.append(f"mAP@0.75:      {map75:.4f}")
    lines.append(f"Mean Precision: {mean_precision:.4f}")
    lines.append(f"Mean Recall:    {mean_recall:.4f}")
    lines.append(f"Mean F1 Score:  {mean_f1:.4f}")
    lines.append("")

    # YOLO per-class mAP metrics
    if hasattr(yolo_precision, "__iter__") and len(yolo_precision) == NUM_CLASSES:
        lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        lines.append("-" * 55)
        for i, name in enumerate(CLASS_NAMES):
            p_i = float(yolo_precision[i])
            r_i = float(yolo_recall[i])
            f1_i = float(yolo_f1[i])
            lines.append(f"{name:<20} {p_i:>10.4f} {r_i:>10.4f} {f1_i:>10.4f}")
        lines.append("")
    lines.append("")

    # IoU-matched classification metrics (DETR-style)
    lines.append("OVERALL STATISTICS (IoU-matched, DETR-style)")
    lines.append("-" * 40)

    if len(filtered_true) > 0:
        overall_accuracy = accuracy_score(filtered_true, filtered_pred)
        overall_precision = precision_score(filtered_true, filtered_pred, average="macro", zero_division=0)
        overall_recall = recall_score(filtered_true, filtered_pred, average="macro", zero_division=0)
        overall_f1 = f1_score(filtered_true, filtered_pred, average="macro", zero_division=0)

        lines.append(f"Accuracy:  {overall_accuracy:.4f}")
        lines.append(f"Precision: {overall_precision:.4f} (macro avg)")
        lines.append(f"Recall:    {overall_recall:.4f} (macro avg)")
        lines.append(f"F1 Score:  {overall_f1:.4f} (macro avg)")
    else:
        lines.append("No predictions to evaluate.")

    lines.append("")
    lines.append("")

    # Per-class classification metrics
    lines.append("PER-CLASS STATISTICS (IoU-matched)")
    lines.append("-" * 70)
    lines.append(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 70)

    if len(filtered_true) > 0:
        report_dict = classification_report(
            filtered_true,
            filtered_pred,
            labels=all_class_names,
            output_dict=True,
            zero_division=0,
        )
        for class_name in all_class_names:
            if class_name in report_dict:
                stats = report_dict[class_name]
                lines.append(
                    f"{class_name:<25} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
                    f"{stats['f1-score']:>10.4f} {stats['support']:>10.0f}"
                )

    lines.append("-" * 70)
    lines.append("")
    lines.append("")

    # Detection summary
    lines.append("DETECTION SUMMARY")
    lines.append("-" * 40)
    tp = sum(1 for t, p in zip(all_true_names, all_pred_names) if t == p and t != "background")
    fp = sum(1 for t, p in zip(all_true_names, all_pred_names) if t == "background" and p != "background")
    fn = sum(1 for t, p in zip(all_true_names, all_pred_names) if t != "background" and p == "background")
    misclass = sum(1 for t, p in zip(all_true_names, all_pred_names) if t != p and t != "background" and p != "background")

    lines.append(f"True Positives:     {tp}")
    lines.append(f"False Positives:    {fp}")
    lines.append(f"False Negatives:    {fn}")
    lines.append(f"Misclassified:      {misclass}")
    lines.append(f"Total Predictions:  {sum(1 for p in all_pred_names if p != 'background')}")
    lines.append(f"Total Ground Truth: {sum(1 for t in all_true_names if t != 'background')}")
    lines.append("")
    lines.append("")

    # Latency statistics
    lines.append("LATENCY STATISTICS (per image)")
    lines.append("-" * 40)
    lines.append(f"Mean:     {latency_mean:.2f} ms")
    lines.append(f"Std Dev:  {latency_std:.2f} ms")
    lines.append(f"Median:   {latency_median:.2f} ms")
    lines.append(f"P95:      {latency_p95:.2f} ms")
    lines.append(f"Min:      {latency_min:.2f} ms")
    lines.append(f"Max:      {latency_max:.2f} ms")
    lines.append(f"Total Inference Time: {float(np.sum(latencies_np)):.2f} s")
    lines.append(f"Throughput: {len(test_images) / float(np.sum(latencies_np)):.1f} images/s")
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    with open(results_path, "w") as f:
        f.write(report_text)
    print(f"\nResults saved to {results_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("CAR DAMAGE DETECTION - YOLOv8 Pipeline")
    print("=" * 60)

    # ── Preprocessing is commented out ──
    # To re-enable, uncomment the block below and change the yaml/eval
    # paths from TRAIN_IMG_DIR/TEST_IMG_DIR back to the preprocessed dirs.
    #
    # print("\n[Step 1/4] Preprocessing training images...")
    # preprocessed_train = PROJECT_ROOT / "train_preprocessed" / "images"
    # preprocess_dataset(TRAIN_IMG_DIR, preprocessed_train)
    #
    # preprocessed_train_labels = PROJECT_ROOT / "train_preprocessed" / "labels"
    # preprocessed_train_labels.mkdir(parents=True, exist_ok=True)
    # for lbl in TRAIN_LBL_DIR.iterdir():
    #     img_stem = lbl.stem
    #     matching_imgs = [f for f in preprocessed_train.iterdir() if f.stem == img_stem]
    #     if matching_imgs:
    #         import shutil
    #         shutil.copy2(str(lbl), str(preprocessed_train_labels / lbl.name))
    #
    # print("\n[Step 2/4] Preprocessing test images...")
    # preprocessed_test = PROJECT_ROOT / "test_preprocessed" / "images"
    # preprocess_dataset(TEST_IMG_DIR, preprocessed_test)
    #
    # preprocessed_test_labels = PROJECT_ROOT / "test_preprocessed" / "labels"
    # preprocessed_test_labels.mkdir(parents=True, exist_ok=True)
    # for lbl in TEST_LBL_DIR.iterdir():
    #     img_stem = lbl.stem
    #     matching_imgs = [f for f in preprocessed_test.iterdir() if f.stem == img_stem]
    #     if matching_imgs:
    #         import shutil
    #         shutil.copy2(str(lbl), str(preprocessed_test_labels / lbl.name))

    # ── Create dataset YAML (using raw train/test dirs directly) ──
    print("\n[Step 1/2] Training YOLOv8 model...")
    yaml_path = create_dataset_yaml(TRAIN_IMG_DIR, TEST_IMG_DIR)

    model = YOLO("yolov8m.pt")

    train_start = time.time()
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=16 if torch.cuda.is_available() else 16,
        imgsz=IMG_SIZE,
        device="0" if torch.cuda.is_available() else "cpu",
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.1,
        patience=30,
        save=True,
        save_period=-1,
        pretrained=True,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.1,
    )
    train_end = time.time()
    training_time = train_end - train_start

    # Save best model
    model.save(str(MODEL_SAVE_PATH))
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # ── Evaluate ──
    print("\n[Step 2/2] Evaluating model...")
    evaluate_and_write_results(
        model=model,
        yaml_path=yaml_path,
        test_img_dir=TEST_IMG_DIR,
        test_lbl_dir=TEST_LBL_DIR,
        results_path=RESULTS_FILE,
        training_time=training_time,
    )


if __name__ == "__main__":
    main()
