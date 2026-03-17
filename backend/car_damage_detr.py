"""
Car Parts Detection using DETR (DEtection TRansformer)
======================================================
Fine-tunes a pretrained DETR-ResNet-50 model on the car parts dataset
for object detection of 21 car part classes.

Pipeline:
1. Data loading and JSON annotation parsing
2. Preprocessing (normalization, angle correction, quality validation)
3. Custom Dataset class for DETR
4. Fine-tuning the pretrained model
5. Evaluation with per-class and overall metrics
"""

import json
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS BEFORE RUNNING
# =============================================================================
# Remember: the folder named "Car_damages_dataset" actually contains PART labels
CURR_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURR_DIR.parent
PARTS_ROOT = os.path.join(PARENT_DIR, "Car parts dataset/File1")
IMG_DIR = os.path.join(PARTS_ROOT, "img")
ANN_DIR = os.path.join(PARTS_ROOT, "ann")

# Training hyperparameters
BATCH_SIZE = 2  # Reduced for GPU memory with unfrozen backbone
LEARNING_RATE_BACKBONE = 2e-5  # Lower LR for pretrained backbone
LEARNING_RATE_HEAD = 2e-4  # Higher LR for randomly initialized head
NUM_EPOCHS = 50
SUBSET_SIZE = 500  # None = use full dataset
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 8  # More patience for full training
TEST_SPLIT = 0.2
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# STEP 1: BUILD METADATA DATAFRAME
# =============================================================================
def build_metadata_df(img_dir, ann_dir):
    """
    Build a DataFrame mapping each image filename to its JSON annotation path.
    Also discovers all unique class names in the dataset.
    """
    image_files = sorted(os.listdir(img_dir))
    records = []
    for img_name in image_files:
        json_name = img_name + ".json"
        json_path = os.path.join(ann_dir, json_name)
        if os.path.exists(json_path):
            records.append({"image_path": img_name, "json_path": json_name})
        else:
            print(f"WARNING: No annotation found for {img_name}, skipping.")
    df = pd.DataFrame(records)
    print(f"Built metadata DataFrame with {len(df)} samples.")
    return df


def discover_classes(ann_dir):
    """
    Scan all JSON files to find every unique classTitle.
    Returns:
        class_to_id: dict mapping class name -> integer label (1-indexed, 0 = background)
        id_to_class: dict mapping integer label -> class name
    """
    unique_classes = set()
    for json_file in os.listdir(ann_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(ann_dir, json_file), "r") as f:
                data = json.load(f)
            for obj in data["objects"]:
                unique_classes.add(obj["classTitle"])

    # Sort for reproducibility; reserve 0 for background / no-object
    sorted_classes = sorted(unique_classes)
    class_to_id = {name: idx + 1 for idx, name in enumerate(sorted_classes)}
    id_to_class = {idx + 1: name for idx, name in enumerate(sorted_classes)}
    id_to_class[0] = "background"

    print(f"Discovered {len(sorted_classes)} part classes: {sorted_classes}")
    return class_to_id, id_to_class


# =============================================================================
# STEP 2: ANNOTATION PARSING
# =============================================================================
def parse_annotation(json_name, ann_dir):
    """
    Parse a single JSON annotation file.
    Returns a dict with:
        - "size": (height, width)
        - "annotations": list of {"class": str, "polygon": [[x,y], ...], "bbox": [x_min, y_min, x_max, y_max]}
    """
    json_path = os.path.join(ann_dir, json_name)
    with open(json_path, "r") as f:
        data = json.load(f)

    h = data["size"]["height"]
    w = data["size"]["width"]

    annotations = []
    for obj in data["objects"]:
        polygon = obj["points"]["exterior"]
        polygon_np = np.array(polygon)

        # Convert polygon to bounding box [x_min, y_min, x_max, y_max]
        x_min = float(polygon_np[:, 0].min())
        y_min = float(polygon_np[:, 1].min())
        x_max = float(polygon_np[:, 0].max())
        y_max = float(polygon_np[:, 1].max())

        annotations.append(
            {
                "class": obj["classTitle"],
                "polygon": polygon,
                "bbox": [x_min, y_min, x_max, y_max],
            }
        )

    return {"size": (h, w), "annotations": annotations}


# =============================================================================
# STEP 3: IMAGE PREPROCESSING
# =============================================================================
def validate_image_quality(image_np, min_size=50, min_std=5.0):
    """
    Image quality validation gate.
    Checks:
        - Image is not None (loaded successfully)
        - Image dimensions meet minimum size
        - Image is not nearly uniform (blank or corrupted)
    Returns:
        (is_valid, reason)
    """
    if image_np is None:
        return False, "Image failed to load"
    h, w = image_np.shape[:2]
    if h < min_size or w < min_size:
        return False, f"Image too small: {w}x{h}"
    if image_np.std() < min_std:
        return False, f"Image appears blank or corrupted (std={image_np.std():.2f})"
    return True, "OK"


def correct_angle(image_np):
    """
    Angle correction using edge detection and Hough line transform.
    Detects dominant lines in the image and rotates to correct small skews.
    Only corrects angles within +/- 15 degrees to avoid over-rotating.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image_np  # No lines detected, return original

    # Compute angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Find the median angle (robust to outliers)
    median_angle = np.median(angles)

    # Only correct if the skew is small (within +/- 15 degrees)
    if abs(median_angle) > 15:
        return image_np

    # Rotate image to correct the skew
    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(image_np, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return corrected


def normalize_image(image_np):
    """
    Normalize pixel values to [0, 1] range and convert to float32.
    """
    return image_np.astype(np.float32) / 255.0


# =============================================================================
# STEP 4: DATASET CLASS
# =============================================================================
class CarPartsDataset(Dataset):
    """
    PyTorch Dataset for car parts object detection with DETR.

    For each sample, loads the image, applies preprocessing, and prepares
    DETR-compatible targets (bounding boxes in COCO format + class labels).
    """

    def __init__(self, dataframe, img_dir, ann_dir, class_to_id, processor, apply_preprocessing=True):
        """
        Args:
            dataframe: DataFrame with 'image_path' and 'json_path' columns
            img_dir: path to image folder
            ann_dir: path to annotation folder
            class_to_id: dict mapping class name -> integer ID
            processor: DetrImageProcessor instance
            apply_preprocessing: whether to run quality check + angle correction
        """
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.class_to_id = class_to_id
        self.processor = processor
        self.apply_preprocessing = apply_preprocessing

        # Pre-parse all annotations so we can filter bad images upfront
        self.samples = []
        skipped = 0
        for idx, row in self.df.iterrows():
            img_path = os.path.join(img_dir, row["image_path"])
            image_np = cv2.imread(img_path)
            if image_np is None:
                skipped += 1
                continue
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Quality validation gate
            if self.apply_preprocessing:
                is_valid, reason = validate_image_quality(image_np)
                if not is_valid:
                    print(f"  Skipping {row['image_path']}: {reason}")
                    skipped += 1
                    continue

            # Parse annotations
            parsed = parse_annotation(row["json_path"], ann_dir)

            # Filter out annotations with invalid bboxes
            valid_annotations = []
            for ann in parsed["annotations"]:
                x_min, y_min, x_max, y_max = ann["bbox"]
                if x_max > x_min and y_max > y_min:
                    valid_annotations.append(ann)

            if len(valid_annotations) == 0:
                skipped += 1
                continue

            self.samples.append(
                {
                    "image_path": row["image_path"],
                    "json_path": row["json_path"],
                    "annotations": valid_annotations,
                    "size": parsed["size"],
                }
            )

        print(f"  Dataset initialized: {len(self.samples)} valid samples, {skipped} skipped.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = os.path.join(self.img_dir, sample["image_path"])
        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Preprocessing
        if self.apply_preprocessing:
            image_np = correct_angle(image_np)

        # Convert to PIL for the DETR processor
        image_pil = Image.fromarray(image_np)
        w, h = image_pil.size

        # Build COCO-format annotations list
        # COCO format expects: [x_min, y_min, width, height] in absolute pixels
        coco_annotations = []
        for i, ann in enumerate(sample["annotations"]):
            x_min, y_min, x_max, y_max = ann["bbox"]
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            coco_annotations.append(
                {
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "category_id": self.class_to_id[ann["class"]],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0,
                    "image_id": idx,
                    "id": i,
                }
            )

        # Wrap in the format the DETR processor expects
        target = {
            "image_id": idx,
            "annotations": coco_annotations,
        }

        # Process image through DETR processor (handles resizing, normalization internally)
        encoding = self.processor(
            images=image_pil,
            annotations=[target],
            return_tensors="pt",
        )

        # Remove batch dimension added by the processor
        pixel_values = encoding["pixel_values"].squeeze(0)
        pixel_mask = encoding["pixel_mask"].squeeze(0)
        labels_out = encoding["labels"][0]  # dict with boxes, class_labels, etc.

        return pixel_values, pixel_mask, labels_out


def collate_fn(batch):
    """
    Custom collate function for DETR.
    Pads images and masks to the largest size in the batch since
    different images may have different aspect ratios after processing.
    """
    # Find max height and width in this batch
    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)

    padded_pixel_values = []
    padded_pixel_masks = []
    labels = []

    for pixel_values, pixel_mask, label in batch:
        c, h, w = pixel_values.shape
        # Pad pixel_values (C, H, W) with zeros on the right and bottom
        padded_img = torch.zeros(c, max_h, max_w, dtype=pixel_values.dtype)
        padded_img[:, :h, :w] = pixel_values
        padded_pixel_values.append(padded_img)

        # Pad pixel_mask (H, W) with zeros (0 = padding)
        padded_mask = torch.zeros(max_h, max_w, dtype=pixel_mask.dtype)
        padded_mask[:h, :w] = pixel_mask
        padded_pixel_masks.append(padded_mask)

        labels.append(label)

    return torch.stack(padded_pixel_values), torch.stack(padded_pixel_masks), labels


# =============================================================================
# STEP 5: TRAINING
# =============================================================================
def train_one_epoch(model, dataloader, optimizer, device, accumulation_steps=4):
    """
    Run one training epoch with gradient accumulation.
    Effective batch size = BATCH_SIZE * accumulation_steps.
    Returns average loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, (pixel_values, pixel_mask, labels) in enumerate(tqdm(dataloader, desc="  Training")):
        pixel_values = pixel_values.to(device)
        pixel_mask = pixel_mask.to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss / accumulation_steps  # Scale loss by accumulation steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # Unscale for logging
        num_batches += 1

    return total_loss / max(num_batches, 1)


# =============================================================================
# STEP 6: EVALUATION
# =============================================================================
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [x_min, y_min, x_max, y_max] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


def evaluate_model(model, dataloader, processor, device, id_to_class, confidence_threshold, iou_threshold):
    """
    Evaluate the model on the test set.

    For each image, matches predicted boxes to ground truth boxes using IoU.
    Collects matched (predicted_label, true_label) pairs for classification metrics.

    Returns:
        all_pred_labels: list of predicted class IDs
        all_true_labels: list of ground truth class IDs
        all_pred_names: list of predicted class names
        all_true_names: list of ground truth class names
    """
    model.eval()
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        for pixel_values, pixel_mask, labels in tqdm(dataloader, desc="  Evaluating"):
            pixel_values = pixel_values.to(device)
            pixel_mask = pixel_mask.to(device)

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # Build target_sizes for the entire batch
            batch_target_sizes = []
            for target in labels:
                orig_size = target["orig_size"].cpu()
                batch_target_sizes.append([orig_size[0].item(), orig_size[1].item()])
            batch_target_sizes = torch.tensor(batch_target_sizes, dtype=torch.long).to(device)

            # Post-process all predictions in the batch at once
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=batch_target_sizes,
                threshold=confidence_threshold,
            )

            # Now match predictions to ground truth per image
            for i, target in enumerate(labels):
                gt_boxes = target["boxes"].cpu().numpy()  # COCO format (cx, cy, w, h) normalized
                gt_labels = target["class_labels"].cpu().numpy()

                h_orig = batch_target_sizes[i][0].item()
                w_orig = batch_target_sizes[i][1].item()

                # Convert GT boxes from COCO normalized to [x_min, y_min, x_max, y_max]
                gt_boxes_xyxy = []
                for box in gt_boxes:
                    cx, cy, bw, bh = box
                    x_min = (cx - bw / 2) * w_orig
                    y_min = (cy - bh / 2) * h_orig
                    x_max = (cx + bw / 2) * w_orig
                    y_max = (cy + bh / 2) * h_orig
                    gt_boxes_xyxy.append([x_min, y_min, x_max, y_max])
                gt_boxes_xyxy = np.array(gt_boxes_xyxy)

                pred_boxes = results[i]["boxes"].cpu().numpy()
                pred_labels_batch = results[i]["labels"].cpu().numpy()
                pred_scores = results[i]["scores"].cpu().numpy()

                # Match predictions to ground truth using IoU
                matched_gt = set()
                for pred_idx in np.argsort(-pred_scores):  # highest confidence first
                    pred_box = pred_boxes[pred_idx]
                    pred_label = pred_labels_batch[pred_idx]
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx in range(len(gt_boxes_xyxy)):
                        if gt_idx in matched_gt:
                            continue
                        iou = compute_iou(pred_box, gt_boxes_xyxy[gt_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        # True positive: matched prediction to GT
                        matched_gt.add(best_gt_idx)
                        all_pred_labels.append(pred_label)
                        all_true_labels.append(gt_labels[best_gt_idx])
                    else:
                        # False positive: prediction with no matching GT
                        all_pred_labels.append(pred_label)
                        all_true_labels.append(0)  # 0 = background (no match)

                # False negatives: GT boxes with no matching prediction
                for gt_idx in range(len(gt_boxes_xyxy)):
                    if gt_idx not in matched_gt:
                        all_pred_labels.append(0)  # predicted nothing
                        all_true_labels.append(gt_labels[gt_idx])

    # Convert to class names
    all_pred_names = [id_to_class.get(l, "background") for l in all_pred_labels]
    all_true_names = [id_to_class.get(l, "background") for l in all_true_labels]

    return all_pred_labels, all_true_labels, all_pred_names, all_true_names


def write_results(
    all_pred_names, all_true_names, id_to_class, output_path
):
    """
    Write overall and per-class classification metrics to a txt file.
    """
    # Get all class names that appear in predictions or ground truth (excluding background)
    all_class_names = sorted(
        set(all_true_names + all_pred_names) - {"background"}
    )

    lines = []
    lines.append("=" * 70)
    lines.append("CAR PARTS DETECTION - EVALUATION RESULTS (DETR)")
    lines.append("=" * 70)
    lines.append("")

    # Overall metrics (excluding background-vs-background matches)
    # Filter to only cases where at least one side is not background
    filtered_true = []
    filtered_pred = []
    for t, p in zip(all_true_names, all_pred_names):
        if t != "background" or p != "background":
            filtered_true.append(t)
            filtered_pred.append(p)

    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)

    if len(filtered_true) > 0:
        overall_accuracy = accuracy_score(filtered_true, filtered_pred)
        lines.append(f"Accuracy:  {overall_accuracy:.4f}")

        overall_precision = precision_score(
            filtered_true, filtered_pred, average="macro", zero_division=0
        )
        overall_recall = recall_score(
            filtered_true, filtered_pred, average="macro", zero_division=0
        )
        overall_f1 = f1_score(
            filtered_true, filtered_pred, average="macro", zero_division=0
        )

        lines.append(f"Precision: {overall_precision:.4f} (macro avg)")
        lines.append(f"Recall:    {overall_recall:.4f} (macro avg)")
        lines.append(f"F1 Score:  {overall_f1:.4f} (macro avg)")
    else:
        lines.append("No predictions to evaluate.")

    lines.append("")
    lines.append("")

    # Per-class metrics
    lines.append("PER-CLASS STATISTICS")
    lines.append("-" * 70)
    lines.append(
        f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    )
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

    # Confusion-style summary
    lines.append("DETECTION SUMMARY")
    lines.append("-" * 40)
    tp = sum(1 for t, p in zip(all_true_names, all_pred_names) if t == p and t != "background")
    fp = sum(1 for t, p in zip(all_true_names, all_pred_names) if t == "background" and p != "background")
    fn = sum(1 for t, p in zip(all_true_names, all_pred_names) if t != "background" and p == "background")
    misclass = sum(
        1
        for t, p in zip(all_true_names, all_pred_names)
        if t != p and t != "background" and p != "background"
    )
    lines.append(f"True Positives (correct class):  {tp}")
    lines.append(f"False Positives (no GT match):   {fp}")
    lines.append(f"False Negatives (missed GT):     {fn}")
    lines.append(f"Misclassifications:              {misclass}")
    lines.append(f"Total GT objects:                {tp + fn + misclass}")
    lines.append(f"Total predictions:               {tp + fp + misclass}")

    # Write to file
    output_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(output_text)

    print(f"\nResults written to: {output_path}")
    print(output_text)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Car Parts Detection with DETR")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # ---- Step 1: Build metadata ----
    print("\n[1/6] Building metadata DataFrame...")
    parts_df = build_metadata_df(IMG_DIR, ANN_DIR)
    class_to_id, id_to_class = discover_classes(ANN_DIR)
    num_classes = len(class_to_id) + 1  # +1 for background (class 0)

    # ---- Step 2: Train/test split (80/20) ----
    print("\n[2/6] Splitting into train/test (80/20)...")

    # Optionally reduce dataset size for faster experimentation
    if SUBSET_SIZE is not None and SUBSET_SIZE < len(parts_df):
        print(f"  Subsetting to {SUBSET_SIZE} images (from {len(parts_df)})...")
        parts_df = parts_df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df, test_df = train_test_split(
        parts_df, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(train_df)} images")
    print(f"  Test:  {len(test_df)} images")

    # ---- Step 3: Initialize DETR processor and model ----
    print("\n[3/6] Loading pretrained DETR-ResNet-50...")
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        revision="no_timm",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)
    print(f"  Model loaded with {num_classes} output classes (including background).")

    # All parameters are trainable (backbone unfrozen for full fine-tuning)
    # Use differential learning rates: lower for pretrained backbone, higher for head
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"  Full fine-tuning: all {trainable_params:,} parameters are trainable.")

    # ---- Step 4: Create datasets and dataloaders ----
    print("\n[4/6] Building datasets...")
    print("  Creating training dataset...")
    train_dataset = CarPartsDataset(
        dataframe=train_df,
        img_dir=IMG_DIR,
        ann_dir=ANN_DIR,
        class_to_id=class_to_id,
        processor=processor,
        apply_preprocessing=True,
    )

    print("  Creating test dataset...")
    test_dataset = CarPartsDataset(
        dataframe=test_df,
        img_dir=IMG_DIR,
        ann_dir=ANN_DIR,
        class_to_id=class_to_id,
        processor=processor,
        apply_preprocessing=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ---- Step 5: Training loop ----
    print(f"\n[5/6] Training for up to {NUM_EPOCHS} epochs (early stopping patience={EARLY_STOPPING_PATIENCE})...")

    # Differential learning rates: backbone gets lower LR to preserve pretrained features,
    # head/decoder gets higher LR since the classification head is randomly initialized
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE},
            {"params": head_params, "lr": LEARNING_RATE_HEAD},
        ],
        weight_decay=1e-4,
    )
    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,} (lr={LEARNING_RATE_BACKBONE})")
    print(f"  Head params:     {sum(p.numel() for p in head_params):,} (lr={LEARNING_RATE_HEAD})")

    # Cosine annealing gradually reduces LR to near zero, better than step decay for long training
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        lr_scheduler.step()
        print(f"  Average loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # Save best model checkpoint
            save_path = os.path.join(PARENT_DIR, "detr_car_damage_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved (loss={best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs.")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs.")
                break

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(PARENT_DIR, "detr_car_damage_best.pt")))

    # ---- Step 6: Evaluation ----
    print(f"\n[6/6] Evaluating on test set ({len(test_dataset)} images)...")
    all_pred_labels, all_true_labels, all_pred_names, all_true_names = evaluate_model(
        model=model,
        dataloader=test_loader,
        processor=processor,
        device=DEVICE,
        id_to_class=id_to_class,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )

    # Write results
    results_path = os.path.join(PARENT_DIR, "car_damage_detection_results.txt")
    write_results(all_pred_names, all_true_names, id_to_class, results_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
