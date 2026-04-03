"""
Car Damage Detection — EfficientDet-D0
=======================================
Converted from Colab notebook to standalone Python script.

Dataset : CarDD (COCO format) — https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection
Model   : EfficientDet-D0 pretrained on COCO, fine-tuned on 6 damage classes
Classes : dent, scratch, crack, glass shatter, lamp broken, tire flat
Runtime : ~1.5–2 hours on a T4 GPU

Usage
-----
  # First download the dataset (requires kaggle.json configured):
  python car_damage_efficientdet.py

  # Override defaults via CLI flags:
  python car_damage_efficientdet.py --epochs 20 --batch_size 4 --output ./runs/exp1
"""

# =============================================================================
# 0. Dependencies — install via:
#    pip install torch torchvision effdet timm albumentations kagglehub tqdm
#    pip install scikit-learn matplotlib opencv-python-headless
# =============================================================================

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from effdet import create_model_from_config, get_efficientdet_config
from effdet.anchors import Anchors, AnchorLabeler, generate_detections
from effdet.loss import DetectionLoss


# =============================================================================
# 1. Configuration
# =============================================================================

def get_config(args=None):
    cfg = dict(
        BATCH_SIZE=8,
        LEARNING_RATE_BACKBONE=1e-4,
        LEARNING_RATE_HEAD=5e-4,
        NUM_EPOCHS=30,
        SUBSET_SIZE=None,          # None = use all images
        CONFIDENCE_THRESHOLD=0.3,
        IOU_THRESHOLD=0.5,
        EARLY_STOPPING_PATIENCE=7,
        RANDOM_SEED=42,
        IMAGE_SIZE=512,
        NUM_WORKERS=2,
        GRAD_ACCUMULATION_STEPS=2,
        OUTPUT_DIR="./output",
    )
    if args:
        for k, v in vars(args).items():
            key = k.upper()
            if key in cfg and v is not None:
                cfg[key] = v
    cfg["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="EfficientDet-D0 Car Damage Detection")
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--epochs",        type=int,   default=None, dest="NUM_EPOCHS")
    p.add_argument("--lr_backbone",   type=float, default=None, dest="LEARNING_RATE_BACKBONE")
    p.add_argument("--lr_head",       type=float, default=None, dest="LEARNING_RATE_HEAD")
    p.add_argument("--subset_size",   type=int,   default=None, dest="SUBSET_SIZE")
    p.add_argument("--conf_thresh",   type=float, default=None, dest="CONFIDENCE_THRESHOLD")
    p.add_argument("--patience",      type=int,   default=None, dest="EARLY_STOPPING_PATIENCE")
    p.add_argument("--num_workers",   type=int,   default=None, dest="NUM_WORKERS")
    p.add_argument("--output",        type=str,   default=None, dest="OUTPUT_DIR")
    return p.parse_args()


# =============================================================================
# 2. Dataset download & path resolution
# =============================================================================

def setup_kaggle_credentials():
    """Prompt for Kaggle credentials if not already present."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Kaggle credentials not found.")
        print('Get yours at https://www.kaggle.com/settings → API → Create New Token')
        print('Paste the full JSON contents ({"username":"...","key":"..."}):\n')
        creds = input("kaggle.json contents: ").strip()
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(creds)
        os.chmod(str(kaggle_json), 0o600)
        print("Credentials saved.\n")
    else:
        print("Kaggle credentials found.\n")


def download_dataset():
    import kagglehub
    print("Downloading CarDD dataset (~5.6 GB) — may take a few minutes...")
    dataset_path = kagglehub.dataset_download("nasimetemadi/car-damage-detection")
    print(f"Dataset at: {dataset_path}\n")
    return Path(dataset_path)


def resolve_paths(dataset_root: Path) -> dict:
    coco_candidates = list(dataset_root.rglob("CarDD_COCO"))
    if coco_candidates:
        coco_dir = coco_candidates[0]
    else:
        ann_candidates = list(dataset_root.rglob("instances_train2017.json"))
        if not ann_candidates:
            raise FileNotFoundError("Could not find COCO annotation files in the dataset.")
        coco_dir = ann_candidates[0].parent.parent

    paths = {
        "train_images": str(coco_dir / "train2017"),
        "val_images":   str(coco_dir / "val2017"),
        "test_images":  str(coco_dir / "test2017"),
        "train_ann":    str(coco_dir / "annotations" / "instances_train2017.json"),
        "val_ann":      str(coco_dir / "annotations" / "instances_val2017.json"),
        "test_ann":     str(coco_dir / "annotations" / "instances_test2017.json"),
    }

    print("Dataset paths:")
    for k, v in paths.items():
        exists = os.path.exists(v)
        count = len(os.listdir(v)) if exists and os.path.isdir(v) else "-"
        status = f"[OK] {count} files" if exists else "[MISSING]"
        print(f"  {k}: {v}  {status}")
    print()
    return paths


# =============================================================================
# 3. COCO Annotation Loader
# =============================================================================

def load_coco_annotations(ann_path: str):
    """Load COCO JSON → (images_info, annotations_by_image_id, categories)."""
    with open(ann_path, "r") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images_info = {
        img["id"]: {"file_name": img["file_name"], "width": img["width"], "height": img["height"]}
        for img in coco["images"]
    }
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        annotations[ann["image_id"]].append({
            "bbox": [x, y, x + w, y + h],
            "category_id": ann["category_id"],
        })

    n_anns = sum(len(v) for v in annotations.values())
    print(f"  {Path(ann_path).name}: {len(images_info)} images, {n_anns} annotations")
    return images_info, dict(annotations), categories


# =============================================================================
# 4. Transforms & Dataset
# =============================================================================

def validate_image_quality(image_np, min_size=50, min_std=5.0):
    if image_np is None:
        return False, "Load failed"
    h, w = image_np.shape[:2]
    if h < min_size or w < min_size:
        return False, f"Too small: {w}x{h}"
    if image_np.std() < min_std:
        return False, f"Blank (std={image_np.std():.1f})"
    return True, "OK"


def correct_angle(image_np):
    """Correct small skew via Hough lines (within ±15 degrees)."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return image_np
    angles = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines]
    median_angle = np.median(angles)
    if abs(median_angle) > 15:
        return image_np
    h, w = image_np.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(image_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def get_train_transforms(image_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.GaussNoise(p=0.2),
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc', label_fields=['labels'],
        min_area=1.0, min_visibility=0.1))


def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc', label_fields=['labels'],
        min_area=1.0, min_visibility=0.1))


class CarDamageDataset(Dataset):
    """COCO-format car damage dataset for EfficientDet."""

    def __init__(self, img_dir, images_info, annotations, transforms=None,
                 apply_preprocessing=True, subset_size=None, image_size=512, random_seed=42):
        self.img_dir = img_dir
        self.transforms = transforms
        self.apply_preprocessing = apply_preprocessing
        self.image_size = image_size
        self.samples = []
        skipped = 0

        image_ids = sorted(images_info.keys())
        if subset_size and subset_size < len(image_ids):
            rng = np.random.RandomState(random_seed)
            image_ids = list(rng.choice(image_ids, size=subset_size, replace=False))

        for img_id in image_ids:
            info = images_info[img_id]
            img_path = os.path.join(img_dir, info["file_name"])
            if not os.path.exists(img_path):
                skipped += 1
                continue
            anns = annotations.get(img_id, [])
            if not anns:
                skipped += 1
                continue

            w, h = info["width"], info["height"]
            valid_anns = []
            for ann in anns:
                x0, y0, x1, y1 = ann["bbox"]
                x0 = max(0.0, min(x0, w - 1))
                y0 = max(0.0, min(y0, h - 1))
                x1 = max(x0 + 1, min(x1, float(w)))
                y1 = max(y0 + 1, min(y1, float(h)))
                if x1 > x0 + 1 and y1 > y0 + 1:
                    valid_anns.append({"bbox": [x0, y0, x1, y1],
                                       "category_id": ann["category_id"]})

            if not valid_anns:
                skipped += 1
                continue
            self.samples.append({
                "file_name": info["file_name"],
                "annotations": valid_anns,
                "width": w, "height": h,
            })

        print(f"    {len(self.samples)} valid, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_np = cv2.imread(os.path.join(self.img_dir, sample["file_name"]))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        if self.apply_preprocessing:
            image_np = correct_angle(image_np)

        h, w = image_np.shape[:2]
        boxes, labels = [], []
        for ann in sample["annotations"]:
            x0, y0, x1, y1 = ann["bbox"]
            boxes.append([max(0, min(x0, w - 1)), max(0, min(y0, h - 1)),
                          max(1, min(x1, float(w))), max(1, min(y1, float(h)))])
            labels.append(ann["category_id"])

        if self.transforms:
            t = self.transforms(image=image_np, bboxes=boxes, labels=labels)
            image_tensor = t["image"]
            boxes = t["bboxes"]
            labels = t["labels"]
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return image_tensor, {
            "boxes": boxes, "labels": labels,
            "img_scale": torch.tensor([1.0]),
            "img_size": torch.tensor([self.image_size, self.image_size]),
        }


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# =============================================================================
# 5. EfficientDet Model
# =============================================================================

class EfficientDetModel(nn.Module):
    def __init__(self, num_classes, image_size=512, arch="tf_efficientdet_d0"):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        config = get_efficientdet_config(arch)
        config.image_size = [image_size, image_size]
        config.num_classes = num_classes
        config.soft_nms = False
        config.max_det_per_image = 100

        config_pretrained = get_efficientdet_config(arch)
        config_pretrained.image_size = [image_size, image_size]
        self.net = create_model_from_config(
            config_pretrained, bench_task='', pretrained=True, num_classes=num_classes)
        self.config = config

        self.anchors = Anchors.from_config(config)
        self.anchor_labeler = AnchorLabeler(self.anchors, num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(config)
        self.num_levels = self.anchors.max_level - self.anchors.min_level + 1

        print("  Model created with pretrained COCO backbone")

    def forward(self, images, targets=None):
        class_out, box_out = self.net(images)

        if self.training and targets is not None:
            bs = images.shape[0]
            all_cls, all_box, all_num_pos = [], [], []

            for i in range(bs):
                gt_boxes = targets[i]["boxes"].to(images.device).float()
                gt_labels = targets[i]["labels"].to(images.device).long()

                if gt_boxes.shape[0] == 0:
                    cls_t, box_t, num_pos = self.anchor_labeler.label_anchors(
                        torch.zeros(0, 4, device=images.device),
                        torch.zeros(0, dtype=torch.long, device=images.device))
                else:
                    gt_boxes_yxyx = torch.stack([
                        gt_boxes[:, 1], gt_boxes[:, 0],
                        gt_boxes[:, 3], gt_boxes[:, 2],
                    ], dim=1)
                    cls_t, box_t, num_pos = self.anchor_labeler.label_anchors(
                        gt_boxes_yxyx, gt_labels)

                all_cls.append(cls_t)
                all_box.append(box_t)
                all_num_pos.append(num_pos)

            cls_targets, box_targets = [], []
            for level_idx in range(self.num_levels):
                cls_targets.append(torch.stack([all_cls[i][level_idx] for i in range(bs)]))
                box_targets.append(torch.stack([all_box[i][level_idx] for i in range(bs)]))

            num_positives = sum(all_num_pos)
            if isinstance(num_positives, int):
                num_positives = torch.tensor(num_positives, dtype=torch.float32,
                                             device=images.device)

            loss, class_loss, box_loss = self.loss_fn(
                class_out, box_out, cls_targets, box_targets, num_positives)
            return {"loss": loss, "class_loss": class_loss, "box_loss": box_loss}
        else:
            return class_out, box_out

    def predict(self, images):
        self.eval()
        with torch.no_grad():
            class_out, box_out = self.net(images)
        results = []
        for i in range(images.shape[0]):
            cls_i = [c[i:i + 1] for c in class_out]
            box_i = [b[i:i + 1] for b in box_out]
            detections = generate_detections(
                cls_i, box_i, self.anchors.boxes,
                indices=None, classes=None,
                img_scale=torch.tensor([1.0], device=images.device),
                img_size=torch.tensor([[self.image_size, self.image_size]], device=images.device),
                max_det_per_image=self.config.max_det_per_image,
                soft_nms=self.config.soft_nms)
            dets = detections[0]
            mask = dets[:, 4] > 0.01
            dets = dets[mask]
            results.append({
                "boxes": dets[:, :4],
                "scores": dets[:, 4],
                "labels": dets[:, 5].int() + 1,
            })
        return results


# =============================================================================
# 6. Predict helper (manual anchor decoding + NMS)
# =============================================================================

def predict_images(model, images, conf_thresh=0.3):
    """Manual predict without generate_detections — more reliable fallback."""
    model.eval()
    with torch.no_grad():
        class_out, box_out = model.net(images)

    cls_list, box_list = [], []
    for cls_level, box_level in zip(class_out, box_out):
        bs, c, h, w = cls_level.shape
        cls_list.append(cls_level.permute(0, 2, 3, 1).reshape(bs, -1, model.num_classes))
        box_list.append(box_level.permute(0, 2, 3, 1).reshape(bs, -1, 4))

    cls_preds = torch.cat(cls_list, dim=1).sigmoid()
    box_preds = torch.cat(box_list, dim=1)
    anchor_boxes = model.anchors.boxes.to(images.device)

    results = []
    for i in range(images.shape[0]):
        scores_per_class, labels = cls_preds[i].max(dim=1)
        mask = scores_per_class > conf_thresh
        if mask.sum() == 0:
            results.append({"boxes": torch.zeros(0, 4),
                             "scores": torch.zeros(0),
                             "labels": torch.zeros(0, dtype=torch.int)})
            continue

        scores = scores_per_class[mask]
        labels_filt = labels[mask] + 1
        box_filt = box_preds[i][mask]
        anchor_filt = anchor_boxes[mask]

        a_y1, a_x1, a_y2, a_x2 = (anchor_filt[:, 0], anchor_filt[:, 1],
                                    anchor_filt[:, 2], anchor_filt[:, 3])
        a_h = a_y2 - a_y1
        a_w = a_x2 - a_x1
        a_cy = a_y1 + 0.5 * a_h
        a_cx = a_x1 + 0.5 * a_w

        dy, dx, dh, dw = box_filt[:, 0], box_filt[:, 1], box_filt[:, 2], box_filt[:, 3]
        cy = dy * a_h + a_cy
        cx = dx * a_w + a_cx
        h = torch.exp(dh) * a_h
        w = torch.exp(dw) * a_w

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        decoded_boxes = torch.stack([x1, y1, x2, y2], dim=1).clamp(min=0, max=model.image_size)

        keep_boxes, keep_scores, keep_labels = [], [], []
        for cls_id in labels_filt.unique():
            cls_mask = labels_filt == cls_id
            c_boxes = decoded_boxes[cls_mask]
            c_scores = scores[cls_mask]
            nms_idx = torch.ops.torchvision.nms(c_boxes, c_scores, 0.5)
            keep_boxes.append(c_boxes[nms_idx])
            keep_scores.append(c_scores[nms_idx])
            keep_labels.append(labels_filt[cls_mask][nms_idx])

        results.append({
            "boxes": torch.cat(keep_boxes) if keep_boxes else torch.zeros(0, 4),
            "scores": torch.cat(keep_scores) if keep_scores else torch.zeros(0),
            "labels": torch.cat(keep_labels).int() if keep_labels else torch.zeros(0, dtype=torch.int),
        })
    return results


# =============================================================================
# 7. Training & Validation
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device, accumulation_steps):
    model.train()
    total_loss, total_cls, total_box, n = 0, 0, 0, 0
    optimizer.zero_grad()
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="  Train", leave=False)):
        images = images.to(device)
        try:
            output = model(images, targets)
        except Exception as e:
            print(f"  [WARN] Batch {batch_idx} error: {e}")
            continue
        loss = output["loss"] / accumulation_steps
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += output["loss"].item()
        total_cls += output["class_loss"].item()
        total_box += output["box_loss"].item()
        n += 1
    return {"loss": total_loss / max(n, 1),
            "cls":  total_cls  / max(n, 1),
            "box":  total_box  / max(n, 1)}


def validate_one_epoch(model, dataloader, device):
    model.train()   # keep BN in train mode for val loss computation
    total_loss, n = 0, 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="  Val", leave=False):
            images = images.to(device)
            try:
                output = model(images, targets)
                if not (torch.isnan(output["loss"]) or torch.isinf(output["loss"])):
                    total_loss += output["loss"].item()
                    n += 1
            except Exception:
                continue
    return total_loss / max(n, 1)


def run_training(model, train_loader, val_loader, cfg):
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and ("backbone" in n or "fpn" in n)],
         "lr": cfg["LEARNING_RATE_BACKBONE"]},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "backbone" not in n and "fpn" not in n],
         "lr": cfg["LEARNING_RATE_HEAD"]},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["NUM_EPOCHS"], eta_min=1e-7)

    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "efficientdet_car_damage_best.pt"

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    print(f"Training for up to {cfg['NUM_EPOCHS']} epochs "
          f"(patience={cfg['EARLY_STOPPING_PATIENCE']})\n")

    for epoch in range(cfg["NUM_EPOCHS"]):
        epoch_start = time.time()
        print(f"Epoch {epoch + 1}/{cfg['NUM_EPOCHS']}")

        train_losses = train_one_epoch(
            model, train_loader, optimizer, cfg["DEVICE"], cfg["GRAD_ACCUMULATION_STEPS"])
        scheduler.step()
        val_loss = validate_one_epoch(model, val_loader, cfg["DEVICE"])

        history["train_loss"].append(train_losses["loss"])
        history["val_loss"].append(val_loss)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        print(f"  Train: {train_losses['loss']:.4f}  "
              f"(cls:{train_losses['cls']:.4f}  box:{train_losses['box']:.4f})")
        print(f"  Val:   {val_loss:.4f}  [{elapsed:.0f}s, total {total_elapsed / 60:.1f}min]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >> Saved best model → {save_path}  (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{cfg['EARLY_STOPPING_PATIENCE']})")
            if epochs_no_improve >= cfg["EARLY_STOPPING_PATIENCE"]:
                print(f"\nEarly stopping at epoch {epoch + 1}!")
                break
        print()

    total_time = time.time() - start_time
    print(f"Training complete!  Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}\n")
    return history, best_val_loss, total_time, str(save_path)


# =============================================================================
# 8. Loss Curve Plot
# =============================================================================

def plot_loss_curve(history, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o", markersize=4)
    plt.plot(history["val_loss"],   label="Val Loss",   marker="s", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EfficientDet-D0 Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "loss_curve.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Loss curve saved → {out}")


# =============================================================================
# 9. Evaluation
# =============================================================================

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
             (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter)
    return inter / max(union, 1e-6)


def evaluate(model, test_loader, test_dataset, id_to_class, cfg):
    print(f"Evaluating on {len(test_dataset)} test images...")
    all_pred, all_true = [], []
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="  Testing"):
            images = images.to(cfg["DEVICE"])
            results = predict_images(model, images, conf_thresh=cfg["CONFIDENCE_THRESHOLD"])

            for i, target in enumerate(targets):
                gt_boxes    = target["boxes"].cpu().numpy()
                gt_labels   = target["labels"].cpu().numpy()
                pred_boxes  = results[i]["boxes"].cpu().numpy()
                pred_scores = results[i]["scores"].cpu().numpy()
                pred_labels = results[i]["labels"].cpu().numpy()

                matched_gt = set()
                for pi in np.argsort(-pred_scores):
                    best_iou, best_gi = 0, -1
                    for gi in range(len(gt_boxes)):
                        if gi in matched_gt:
                            continue
                        iou = compute_iou(pred_boxes[pi], gt_boxes[gi])
                        if iou > best_iou:
                            best_iou, best_gi = iou, gi
                    if best_iou >= cfg["IOU_THRESHOLD"] and best_gi >= 0:
                        matched_gt.add(best_gi)
                        all_pred.append(int(pred_labels[pi]))
                        all_true.append(int(gt_labels[best_gi]))
                    else:
                        all_pred.append(int(pred_labels[pi]))
                        all_true.append(0)
                for gi in range(len(gt_boxes)):
                    if gi not in matched_gt:
                        all_pred.append(0)
                        all_true.append(int(gt_labels[gi]))

    all_pred_names = [id_to_class.get(l, "background") for l in all_pred]
    all_true_names = [id_to_class.get(l, "background") for l in all_true]

    tp  = sum(1 for t, p in zip(all_true_names, all_pred_names) if t == p and t != "background")
    fp  = sum(1 for t, p in zip(all_true_names, all_pred_names) if t == "background" and p != "background")
    fn  = sum(1 for t, p in zip(all_true_names, all_pred_names) if t != "background" and p == "background")
    mis = sum(1 for t, p in zip(all_true_names, all_pred_names) if t != p and t != "background" and p != "background")

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn + mis) if tp + fn + mis > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    print("\n" + "=" * 50)
    print("CAR DAMAGE DETECTION — FINAL RESULTS")
    print("=" * 50)
    print(f"  True Positives  : {tp}")
    print(f"  False Positives : {fp}")
    print(f"  False Negatives : {fn}")
    print(f"  Misclassified   : {mis}")
    print(f"  Total GT objects: {tp + fn + mis}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  Total preds     : {tp + fp + mis}")

    return dict(tp=tp, fp=fp, fn=fn, mis=mis, prec=prec, rec=rec, f1=f1,
                all_true_names=all_true_names, all_pred_names=all_pred_names)


# =============================================================================
# 10. Visualize Sample Detections
# =============================================================================

COLORS = {1: "red", 2: "blue", 3: "green", 4: "orange", 5: "purple", 6: "cyan"}


def visualize_detections(model, test_dataset, id_to_class, cfg, n_samples=6):
    model.eval()
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)
    cols = 3
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten()

    for ax_idx, data_idx in enumerate(indices):
        img_tensor, target = test_dataset[data_idx]
        images = img_tensor.unsqueeze(0).to(cfg["DEVICE"])
        results = predict_images(model, images, conf_thresh=cfg["CONFIDENCE_THRESHOLD"])

        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

        ax = axes[ax_idx]
        ax.imshow(img_np)

        r = results[0]
        for j in range(len(r["scores"])):
            x1, y1, x2, y2 = r["boxes"][j].cpu().numpy()
            lid = r["labels"][j].item()
            color = COLORS.get(lid, "white")
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(x1, y1 - 5,
                    f"{id_to_class.get(lid, '?')} {r['scores'][j]:.2f}",
                    color=color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        for j in range(len(target["boxes"])):
            x1, y1, x2, y2 = target["boxes"][j].numpy()
            lid = target["labels"][j].item()
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                edgecolor="lime", facecolor="none", linestyle="--"))
            ax.text(x2, y2 + 12, f"GT:{id_to_class.get(lid, '?')}",
                    color="lime", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

        ax.set_title(f"Image {data_idx}")
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.suptitle("Predictions (solid) vs Ground Truth (dashed green)", fontsize=14)
    plt.tight_layout()
    out = Path(cfg["OUTPUT_DIR"]) / "sample_detections.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Detection samples saved → {out}")


# =============================================================================
# 11. Save Results Summary
# =============================================================================

def save_results(metrics, history, best_val_loss, total_time, output_dir):
    lines = [
        "CAR DAMAGE DETECTION — EfficientDet-D0 Results",
        f"Epochs trained  : {len(history['train_loss'])}",
        f"Best val loss   : {best_val_loss:.4f}",
        f"Training time   : {total_time / 60:.1f} minutes",
        "",
        f"Precision       : {metrics['prec']:.4f}",
        f"Recall          : {metrics['rec']:.4f}",
        f"F1 Score        : {metrics['f1']:.4f}",
        "",
        f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  Misclass={metrics['mis']}",
    ]
    out = Path(output_dir) / "results.txt"
    out.write_text("\n".join(lines))
    print(f"Results saved → {out}")


# =============================================================================
# 12. Main
# =============================================================================

def main():
    args = parse_args()
    cfg = get_config(args)

    # --- GPU check ---
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected — training will be very slow on CPU.")
    print(f"Device   : {cfg['DEVICE']}\n")

    # --- Data ---
    setup_kaggle_credentials()
    dataset_root = download_dataset()
    paths = resolve_paths(dataset_root)

    print("Loading annotations...")
    train_imgs, train_anns, categories = load_coco_annotations(paths["train_ann"])
    val_imgs,   val_anns,   _          = load_coco_annotations(paths["val_ann"])
    test_imgs,  test_anns,  _          = load_coco_annotations(paths["test_ann"])

    id_to_class = dict(categories)
    id_to_class[0] = "background"
    num_classes = len(categories)
    print(f"\n{num_classes} damage classes:")
    for cid, name in sorted(categories.items()):
        print(f"  {cid}: {name}")
    print()

    print("Building datasets...")
    print("  Train:")
    train_dataset = CarDamageDataset(
        paths["train_images"], train_imgs, train_anns,
        get_train_transforms(cfg["IMAGE_SIZE"]),
        subset_size=cfg["SUBSET_SIZE"],
        image_size=cfg["IMAGE_SIZE"],
        random_seed=cfg["RANDOM_SEED"])
    print("  Val:")
    val_dataset = CarDamageDataset(
        paths["val_images"], val_imgs, val_anns,
        get_val_transforms(cfg["IMAGE_SIZE"]),
        image_size=cfg["IMAGE_SIZE"])
    print("  Test:")
    test_dataset = CarDamageDataset(
        paths["test_images"], test_imgs, test_anns,
        get_val_transforms(cfg["IMAGE_SIZE"]),
        image_size=cfg["IMAGE_SIZE"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg["NUM_WORKERS"],
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                              collate_fn=collate_fn, num_workers=cfg["NUM_WORKERS"],
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                              collate_fn=collate_fn, num_workers=cfg["NUM_WORKERS"],
                              pin_memory=True)
    print(f"\nDataloaders ready: {len(train_loader)} train | "
          f"{len(val_loader)} val | {len(test_loader)} test\n")

    # --- Model ---
    print(f"Building EfficientDet-D0 ({num_classes} classes)...")
    model = EfficientDetModel(num_classes=num_classes, image_size=cfg["IMAGE_SIZE"])
    model.to(cfg["DEVICE"])
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable params: {train_p:,}\n")

    # --- Train ---
    history, best_val_loss, total_time, save_path = run_training(
        model, train_loader, val_loader, cfg)

    plot_loss_curve(history, cfg["OUTPUT_DIR"])

    # --- Evaluate ---
    print(f"Loading best model from {save_path}...")
    model.load_state_dict(torch.load(save_path, map_location=cfg["DEVICE"], weights_only=True))
    metrics = evaluate(model, test_loader, test_dataset, id_to_class, cfg)

    # --- Visualize ---
    visualize_detections(model, test_dataset, id_to_class, cfg)

    # --- Save summary ---
    save_results(metrics, history, best_val_loss, total_time, cfg["OUTPUT_DIR"])
    print(f"\nAll outputs saved to: {cfg['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
