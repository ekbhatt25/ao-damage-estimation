"""
Supervisely-format dataset loader for Mask R-CNN.

The Kaggle dataset ships as per-image JSON files (Supervisely format) with
polygon segmentation. This module converts them to the dict-of-tensors format
that torchvision's Mask R-CNN expects.

Two dataset modes:
  - "parts"  : 21 car-part classes, sourced from "Car damages dataset" folder
  - "damage" : 8 damage-type classes, sourced from "Car parts dataset" folder
"""
import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from .config import (
    PARTS_IMG_DIR, PARTS_ANN_DIR,
    DAMAGE_IMG_DIR, DAMAGE_ANN_DIR,
    PART_LABEL_MAP, DAMAGE_LABEL_MAP,
    TRAIN_VAL_SPLIT, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS,
)
from .preprocess import preprocess_image


# ── Polygon → binary mask ──────────────────────────────────────────────────

def polygon_to_mask(
    points: list[list[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert a list of [x, y] polygon vertices to a binary mask (uint8)."""
    mask_img = Image.new("L", (width, height), 0)
    flat = [coord for xy in points for coord in xy]
    if len(flat) >= 6:  # need at least 3 points
        ImageDraw.Draw(mask_img).polygon(flat, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


# ── Augmentations (image + target together) ────────────────────────────────

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        target: dict,
    ) -> tuple[torch.Tensor, dict]:
        if random.random() < self.p:
            image = TF.hflip(image)
            w = image.shape[-1]
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if target["masks"].numel():
                target["masks"] = target["masks"].flip(-1)
        return image, target


class ColorJitter:
    """Apply random photometric distortions (image only, not masks)."""
    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.05,
    ):
        self.brightness  = brightness
        self.contrast    = contrast
        self.saturation  = saturation
        self.hue         = hue

    def __call__(
        self,
        image: torch.Tensor,
        target: dict,
    ) -> tuple[torch.Tensor, dict]:
        # Convert to PIL for TF transforms, then back
        pil = TF.to_pil_image(image)
        pil = TF.adjust_brightness(pil, 1 + random.uniform(-self.brightness, self.brightness))
        pil = TF.adjust_contrast(pil,   1 + random.uniform(-self.contrast,   self.contrast))
        pil = TF.adjust_saturation(pil, 1 + random.uniform(-self.saturation, self.saturation))
        pil = TF.adjust_hue(pil,        random.uniform(-self.hue, self.hue))
        return TF.to_tensor(pil), target


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_train_transforms() -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────

class CarDataset(Dataset):
    """
    Loads images + Supervisely polygon annotations for either parts or damages.

    Args:
        mode:    "parts" | "damage"
        split:   "train" | "val"
        augment: apply training augmentations
    """

    def __init__(
        self,
        mode: Literal["parts", "damage"],
        split: Literal["train", "val"],
        augment: bool = False,
    ):
        self.mode    = mode
        self.augment = augment
        self.transforms = get_train_transforms() if augment else None

        if mode == "parts":
            self.img_dir   = PARTS_IMG_DIR
            self.ann_dir   = PARTS_ANN_DIR
            self.label_map = PART_LABEL_MAP
        else:
            self.img_dir   = DAMAGE_IMG_DIR
            self.ann_dir   = DAMAGE_ANN_DIR
            self.label_map = DAMAGE_LABEL_MAP

        all_samples = self._discover_samples()
        self.samples = self._split(all_samples, split)

    # ── internal helpers ───────────────────────────────────────────────────

    def _discover_samples(self) -> list[dict]:
        """
        Pair every annotation JSON with its image file.
        Returns list of {"img_path": Path, "ann_path": Path}.
        """
        samples = []
        for ann_path in sorted(self.ann_dir.iterdir()):
            # annotation filename: "Car damages 100.png.json"
            # image filename: "Car damages 100.png" (or .jpg)
            img_name = ann_path.stem          # strip .json → "Car damages 100.png"
            img_path = self.img_dir / img_name
            if not img_path.exists():
                # try alternative extension
                base = Path(img_name).stem    # "Car damages 100"
                for ext in (".jpg", ".jpeg", ".png"):
                    alt = self.img_dir / (base + ext)
                    if alt.exists():
                        img_path = alt
                        break
                else:
                    continue  # skip if no image found
            samples.append({"img_path": img_path, "ann_path": ann_path})
        return samples

    def _split(self, samples: list[dict], split: str) -> list[dict]:
        rng = random.Random(RANDOM_SEED)
        shuffled = samples[:]
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * TRAIN_VAL_SPLIT)
        return shuffled[:n_train] if split == "train" else shuffled[n_train:]

    def _load_annotations(self, ann_path: Path, h: int, w: int) -> dict:
        """
        Parse Supervisely JSON → masks, boxes, labels tensors.
        Skips objects with unknown classTitle or < 3 polygon points.
        """
        data = json.loads(ann_path.read_text())
        masks, boxes, labels = [], [], []

        for obj in data["objects"]:
            class_title = obj["classTitle"]
            label = self.label_map.get(class_title)
            if label is None or label == 0:
                continue  # skip background or unknown
            if obj.get("geometryType") != "polygon":
                continue

            exterior = obj["points"]["exterior"]
            if len(exterior) < 3:
                continue

            mask = polygon_to_mask(exterior, h, w)
            if mask.sum() == 0:
                continue  # degenerate polygon

            ys, xs = np.where(mask)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            if x2 <= x1 or y2 <= y1:
                continue

            masks.append(mask)
            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        if masks:
            masks_t  = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas    = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            masks_t  = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas    = torch.zeros((0,), dtype=torch.float32)

        return {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "masks":    masks_t,
            "area":     areas,
            "iscrowd":  torch.zeros(len(labels), dtype=torch.int64),
        }

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        result = preprocess_image(str(sample["img_path"]))
        img    = result.image  # PIL RGB

        w, h   = img.size
        target = self._load_annotations(sample["ann_path"], h, w)
        target["image_id"] = torch.tensor([idx])

        image_t = TF.to_tensor(img)  # [3, H, W], float32 in [0, 1]

        if self.augment and self.transforms:
            image_t, target = self.transforms(image_t, target)

        return image_t, target

    def get_sample_path(self, idx: int) -> Path:
        return self.samples[idx]["img_path"]


# ── DataLoader factory ─────────────────────────────────────────────────────

def collate_fn(batch):
    """Variable number of objects per image requires custom collation."""
    return tuple(zip(*batch))


def make_loaders(
    mode: Literal["parts", "damage"],
) -> tuple[DataLoader, DataLoader]:
    train_ds = CarDataset(mode, "train", augment=True)
    val_ds   = CarDataset(mode, "val",   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
