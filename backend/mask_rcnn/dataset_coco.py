"""
COCO-format dataset loader for Mask R-CNN.

Used for the CarDD dataset which ships with official train/val splits
in COCO JSON format. Avoids any format conversion — annotations are
consumed natively via pycocotools.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from .config import (
    CARDD_TRAIN_DIR, CARDD_VAL_DIR,
    CARDD_TRAIN_ANN, CARDD_VAL_ANN,
    CARDD_BATCH_SIZE, CARDD_NUM_WORKERS,
)
from .dataset import collate_fn, get_train_transforms
from .preprocess import preprocess_image


class CocoCarDataset(Dataset):
    """
    Loads images and COCO-format annotations for CarDD.

    Args:
        split:   "train" | "val"
        augment: apply training augmentations
    """

    def __init__(self, split: str = "train", augment: bool = False):
        self.split   = split
        self.augment = augment
        self.transforms = get_train_transforms() if augment else None

        ann_file  = CARDD_TRAIN_ANN  if split == "train" else CARDD_VAL_ANN
        self.img_dir = CARDD_TRAIN_DIR if split == "train" else CARDD_VAL_DIR

        self.coco = COCO(str(ann_file))
        # Keep only image IDs that have at least one annotation
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_id   = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]

        result = preprocess_image(str(img_path))
        img    = result.image  # PIL RGB
        w, h   = img.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        masks, boxes, labels = [], [], []
        for ann in anns:
            label = int(ann["category_id"])
            if label == 0:
                continue

            # Decode segmentation → binary mask
            seg = ann["segmentation"]
            if isinstance(seg, list):
                # Polygon format
                rle = coco_mask_utils.frPyObjects(seg, h, w)
                bin_mask = coco_mask_utils.decode(coco_mask_utils.merge(rle))
            else:
                # Already RLE
                bin_mask = coco_mask_utils.decode(seg)

            bin_mask = bin_mask.astype(np.uint8)
            if bin_mask.sum() == 0:
                continue

            ys, xs = np.where(bin_mask)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            if x2 <= x1 or y2 <= y1:
                continue

            masks.append(bin_mask)
            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        if masks:
            masks_t  = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas    = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            crowd    = torch.zeros(len(labels), dtype=torch.int64)
        else:
            masks_t  = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas    = torch.zeros((0,), dtype=torch.float32)
            crowd    = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "masks":    masks_t,
            "area":     areas,
            "iscrowd":  crowd,
            "image_id": torch.tensor([img_id]),
        }

        image_t = TF.to_tensor(img)

        if self.augment and self.transforms:
            image_t, target = self.transforms(image_t, target)

        return image_t, target

    def get_coco(self) -> COCO:
        """Return the underlying COCO object (for evaluation)."""
        return self.coco


def make_cardd_loaders() -> tuple[DataLoader, DataLoader]:
    train_ds = CocoCarDataset("train", augment=True)
    val_ds   = CocoCarDataset("val",   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CARDD_BATCH_SIZE,
        shuffle=True,
        num_workers=CARDD_NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CARDD_NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
