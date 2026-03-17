# CarDD Mask R-CNN Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a fresh Mask R-CNN damage model on the CarDD COCO dataset, evaluate it, and produce a concise training summary with comparison against the old damage model baseline.

**Architecture:** Add a `cardd` mode to the existing pipeline. A new `dataset_coco.py` loader reads native COCO JSON (no Supervisely conversion). Config gets CarDD paths and RTX 4070-tuned hyperparams. All other modules (train, evaluate, run) get `cardd` added to their mode choices.

**Tech Stack:** Python 3.10, PyTorch 2.6.0+cu124, torchvision, pycocotools, PIL

---

## Chunk 1: Data extraction and config

### Task 1: Extract CarDD dataset

**Files:**
- Create: `data/CarDD/` directory tree via shell

- [ ] **Step 1: Extract the zip**

```bash
source .venv/bin/activate
unzip ~/Downloads/cardatacomp.zip -d data/
```

Expected output ends with something like `inflating: data/CarDD_release/CarDD_COCO/...`

- [ ] **Step 2: Verify structure**

```bash
ls data/CarDD_release/CarDD_COCO/
# Expected: annotations  test2017  train2017  val2017
ls data/CarDD_release/CarDD_COCO/annotations/
# Expected: image_info.xlsx  instances_test2017.json  instances_train2017.json  instances_val2017.json
python3 -c "
import json
with open('data/CarDD_release/CarDD_COCO/annotations/instances_train2017.json') as f:
    d = json.load(f)
print('categories:', [(c['id'], c['name']) for c in d['categories']])
print('train images:', len(d['images']))
print('train annotations:', len(d['annotations']))
"
```

Expected: 6 categories with IDs, 2816 images, ~6211 annotations.

- [ ] **Step 3: Count val set**

```bash
python3 -c "
import json
with open('data/CarDD_release/CarDD_COCO/annotations/instances_val2017.json') as f:
    d = json.load(f)
print('val images:', len(d['images']))
print('val annotations:', len(d['annotations']))
"
```

Expected: 810 images, ~1744 annotations.

---

### Task 2: Add CarDD config to config.py

**Files:**
- Modify: `backend/mask_rcnn/config.py`

- [ ] **Step 1: Add CarDD paths, classes, and RTX 4070 hyperparams**

In `config.py`, after the existing `DAMAGE_MODEL_PATH` line, add:

```python
CARDD_DATA_DIR   = DATA_ROOT / "CarDD_release" / "CarDD_COCO"
CARDD_TRAIN_DIR  = CARDD_DATA_DIR / "train2017"
CARDD_VAL_DIR    = CARDD_DATA_DIR / "val2017"
CARDD_TRAIN_ANN  = CARDD_DATA_DIR / "annotations" / "instances_train2017.json"
CARDD_VAL_ANN    = CARDD_DATA_DIR / "annotations" / "instances_val2017.json"
CARDD_MODEL_PATH = MODELS_DIR / "cardd_model.pth"
```

After the existing `DAMAGE_LABEL_MAP` block, add:

```python
# CarDD categories (IDs as they appear in the COCO JSON)
# Category IDs in the JSON are 1-based; model index = category_id directly.
CARDD_CLASSES = [
    "__background__",   # 0
    "dent",             # 1
    "scratch",          # 2
    "crack",            # 3
    "glass shatter",    # 4
    "lamp broken",      # 5
    "tire flat",        # 6
]
NUM_CARDD_CLASSES = len(CARDD_CLASSES)  # 7
CARDD_LABEL_MAP = {c: i for i, c in enumerate(CARDD_CLASSES)}
```

At the bottom of the training section, add a CarDD-specific block:

```python
# ── CarDD / RTX 4070 Laptop overrides ──────────────────────────────────────
CARDD_BATCH_SIZE    = 4    # RTX 4070 has 8 GB VRAM — batch 4 is comfortable
CARDD_NUM_WORKERS   = 4
CARDD_PHASE1_EPOCHS = 10
CARDD_PHASE2_EPOCHS = 40   # more data warrants more fine-tune epochs
CARDD_PHASE1_LR     = 0.005
CARDD_PHASE2_LR     = 0.001
```

- [ ] **Step 2: Verify config imports cleanly**

```bash
source .venv/bin/activate
python3 -c "from backend.mask_rcnn.config import CARDD_CLASSES, CARDD_MODEL_PATH, CARDD_BATCH_SIZE; print(CARDD_CLASSES, CARDD_BATCH_SIZE)"
```

Expected: list of 7 class names, `4`.

---

## Chunk 2: Dataset loader and model

### Task 3: Write COCO dataset loader

**Files:**
- Create: `backend/mask_rcnn/dataset_coco.py`

CarDD is already in COCO format with official train/val splits. No split logic needed — we just read the JSON directly. The loader converts COCO polygon/RLE segmentations to per-instance binary masks.

- [ ] **Step 1: Create `dataset_coco.py`**

```python
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
```

- [ ] **Step 2: Smoke-test the loader**

```bash
source .venv/bin/activate
python3 -c "
from backend.mask_rcnn.dataset_coco import CocoCarDataset
ds = CocoCarDataset('train', augment=False)
print('train len:', len(ds))
img, tgt = ds[0]
print('image shape:', img.shape)
print('n boxes:', len(tgt['boxes']))
print('labels:', tgt['labels'])
ds2 = CocoCarDataset('val', augment=False)
print('val len:', len(ds2))
"
```

Expected: train len ~2816, val len ~810, image shape (3, H, W), labels with values 1-6.

---

### Task 4: Add build_cardd_model to model.py

**Files:**
- Modify: `backend/mask_rcnn/model.py`

- [ ] **Step 1: Add import and factory function**

In `model.py`, add to the imports at the top:
```python
from .config import NUM_PART_CLASSES, NUM_DAMAGE_CLASSES, NUM_CARDD_CLASSES
```
(replace the existing `from .config import NUM_PART_CLASSES, NUM_DAMAGE_CLASSES` line)

After `build_damage_model`, add:
```python
def build_cardd_model(pretrained: bool = True) -> nn.Module:
    """
    Mask R-CNN for CarDD: 6 damage classes + background.
    Uses full 800/1333 resolution (not Jetson-capped) for RTX 4070.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        min_size=800,
        max_size=1333,
    )
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, NUM_CARDD_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CARDD_CLASSES)
    return model
```

- [ ] **Step 2: Verify**

```bash
source .venv/bin/activate
python3 -c "
from backend.mask_rcnn.model import build_cardd_model, count_params
m = build_cardd_model()
print(count_params(m))
"
```

Expected: total ~44M params, all trainable.

---

## Chunk 3: Training and evaluation wiring

### Task 5: Add cardd mode to train.py

**Files:**
- Modify: `backend/mask_rcnn/train.py`

- [ ] **Step 1: Update imports**

Add to the existing imports block:
```python
from .config import (
    PARTS_MODEL_PATH, DAMAGE_MODEL_PATH, CARDD_MODEL_PATH,
    PHASE1_EPOCHS, PHASE1_LR,
    PHASE2_EPOCHS, PHASE2_LR,
    CARDD_PHASE1_EPOCHS, CARDD_PHASE2_EPOCHS,
    CARDD_PHASE1_LR, CARDD_PHASE2_LR,
    LR_MOMENTUM, LR_WEIGHT_DECAY, LR_STEP_SIZE, LR_GAMMA,
)
from .dataset import make_loaders
from .dataset_coco import make_cardd_loaders
from .model import (
    build_parts_model, build_damage_model, build_cardd_model,
    freeze_backbone, unfreeze_all, count_params,
    enable_gradient_checkpointing,
)
```

- [ ] **Step 2: Update the train() function signature and dispatch**

Change the `train()` function to accept `"cardd"` as a valid mode. Replace the model/path/loader selection block at the top of `train()`:

```python
def train(mode: Literal["parts", "damage", "cardd"]) -> Path:
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training mode : {mode}")
    print(f"Device        : {device}")

    if mode == "cardd":
        train_loader, val_loader = make_cardd_loaders()
        model     = build_cardd_model()
        save_path = CARDD_MODEL_PATH
        ph1_epochs, ph1_lr = CARDD_PHASE1_EPOCHS, CARDD_PHASE1_LR
        ph2_epochs, ph2_lr = CARDD_PHASE2_EPOCHS, CARDD_PHASE2_LR
    else:
        train_loader, val_loader = make_loaders(mode)
        model     = build_parts_model() if mode == "parts" else build_damage_model()
        save_path = PARTS_MODEL_PATH    if mode == "parts" else DAMAGE_MODEL_PATH
        ph1_epochs, ph1_lr = PHASE1_EPOCHS, PHASE1_LR
        ph2_epochs, ph2_lr = PHASE2_EPOCHS, PHASE2_LR
```

Then replace all hardcoded `PHASE1_EPOCHS`/`PHASE2_EPOCHS`/`PHASE1_LR`/`PHASE2_LR` references in the function body with the local variables `ph1_epochs`, `ph1_lr`, `ph2_epochs`, `ph2_lr`.

Also remove the `enable_gradient_checkpointing(model)` call inside the `cardd` path — it's only needed on Jetson. The cleanest way is to conditionally call it:

```python
    if mode != "cardd":
        enable_gradient_checkpointing(model)  # memory saving for Jetson; not needed on RTX 4070
```

Finally, update the `__main__` block:
```python
    parser.add_argument("--mode", choices=["parts", "damage", "cardd"], required=True)
```

- [ ] **Step 3: Verify train.py imports cleanly**

```bash
source .venv/bin/activate
python3 -c "from backend.mask_rcnn.train import train; print('train imported OK')"
```

---

### Task 6: Add cardd mode to evaluate.py

**Files:**
- Modify: `backend/mask_rcnn/evaluate.py`

For CarDD evaluation we leverage the official COCO JSON directly as ground truth — cleaner than regenerating it from our dataset loader, and gives us the official splits.

- [ ] **Step 1: Update imports**

Add to existing imports:
```python
from .config import (
    PARTS_MODEL_PATH, DAMAGE_MODEL_PATH, CARDD_MODEL_PATH,
    PART_CLASSES, DAMAGE_CLASSES, CARDD_CLASSES,
    CARDD_VAL_ANN, CARDD_VAL_DIR,
    SCORE_THRESHOLD, NMS_IOU_THRESHOLD,
    LATENCY_WARMUP_RUNS, LATENCY_BENCH_RUNS,
    MODELS_DIR,
)
from .dataset_coco import CocoCarDataset
from .model import build_parts_model, build_damage_model, build_cardd_model, load_checkpoint
```

- [ ] **Step 2: Add evaluate_cardd() function**

Add this function after the existing `evaluate()` function:

```python
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
        for iou_type in ["segm"]:
            ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
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
    # Build a gt_dict compatible with compute_per_class_metrics
    gt_dict = {
        "images":      [{"id": i} for i in val_ds.coco.getImgIds()],
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
```

Note: `benchmark_latency` takes a dataset that must have `__getitem__` returning `(img_t, target)` — `CocoCarDataset` already does this, so it works unchanged.

- [ ] **Step 3: Update evaluate() dispatcher and __main__**

In `evaluate()`, update the mode check:
```python
def evaluate(mode: Literal["parts", "damage", "cardd"]) -> dict:
    if mode == "cardd":
        return evaluate_cardd()
    # ... rest of existing code unchanged
```

Update `__main__`:
```python
    parser.add_argument("--mode", choices=["parts", "damage", "cardd"], required=True)
```

- [ ] **Step 4: Verify evaluate.py imports cleanly**

```bash
source .venv/bin/activate
python3 -c "from backend.mask_rcnn.evaluate import evaluate; print('evaluate imported OK')"
```

---

### Task 7: Add cardd to run_maskrcnn.py

**Files:**
- Modify: `run_maskrcnn.py`

- [ ] **Step 1: Update all mode choices from `["parts", "damage"]` to `["parts", "damage", "cardd"]`**

There are 4 subparsers that need updating: `train`, `eval`, `run`, and the docstring. Update each `choices=["parts", "damage"]` to `choices=["parts", "damage", "cardd"]` and update the docstring at the top.

- [ ] **Step 2: Verify CLI parses cardd**

```bash
source .venv/bin/activate
python run_maskrcnn.py train --mode cardd --help 2>&1 | head -5
```

Expected: no errors.

---

## Chunk 4: Training run

### Task 8: Run training

- [ ] **Step 1: Confirm GPU is available**

```bash
source .venv/bin/activate
python3 -c "import torch; print(torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')"
```

Expected: RTX 4070, ~8 GB.

- [ ] **Step 2: Launch training (supervise output)**

```bash
source .venv/bin/activate
python run_maskrcnn.py train --mode cardd 2>&1 | tee models/cardd_train.log
```

Monitor for:
- Phase 1 starting (epoch 1/10)
- Loss values decreasing over epochs
- Phase 2 starting (epoch 1/40)
- `↳ new best loss → checkpoint saved` lines appearing
- Final `Checkpoint saved → models/cardd_model.pth`

If OOM error occurs during Phase 2: reduce `CARDD_BATCH_SIZE` to 2 in config.py and restart.
If NaN loss appears: likely a bad image; check preprocess quality gate is catching it.

- [ ] **Step 3: Verify checkpoint saved**

```bash
ls -lh models/cardd_model.pth models/cardd_model.history.json
```

Expected: `cardd_model.pth` ~169 MB, `cardd_model.history.json` present.

---

## Chunk 5: Evaluation and summary

### Task 9: Run evaluation

- [ ] **Step 1: Run eval**

```bash
source .venv/bin/activate
python run_maskrcnn.py eval --mode cardd 2>&1 | tee models/cardd_eval.log
```

Watch for the mAP summary table at the end.

- [ ] **Step 2: Verify results file**

```bash
python3 -c "
import json
r = json.load(open('models/cardd_eval_results.json'))
print('mAP:', r['map']['mAP'])
print('mAP@50:', r['map']['mAP_50'])
for c in r['per_class']:
    print(f\"  {c['class']:<16} F1={c['f1']:.3f}\")
"
```

---

### Task 10: Write CARDD_TRAINING_SUMMARY.md

**Files:**
- Create: `CARDD_TRAINING_SUMMARY.md`

After eval results are in hand, write a concise summary following this structure:

```markdown
# CarDD Mask R-CNN Training Summary

## Platform

| | |
|---|---|
| Device | ASUS ROG Flow X16 (2023) — RTX 4070 Laptop |
| CUDA | 12.4 |
| Python | 3.10.x |
| PyTorch | 2.6.0+cu124 |

## Dataset — CarDD

| | |
|---|---|
| Source | CarDD (COCO format) |
| Train images | 2816 |
| Val images | 810 |
| Classes | 6: dent, scratch, crack, glass shatter, lamp broken, tire flat |
| Split | Official pre-split (used as-is) |

## Training Configuration

| | |
|---|---|
| Architecture | Mask R-CNN ResNet-50-FPN (COCO pretrained) |
| Phase 1 | 10 epochs, backbone frozen, lr=0.005 |
| Phase 2 | 40 epochs, full fine-tune, lr=0.001 (CosineAnnealingLR) |
| Batch size | 4 |
| Image size | 800–1333 px |
| Mixed precision | AMP |

---

## CarDD Model Results

### mAP

| Metric | Value |
|---|---|
| mAP (0.50:0.95) | **X.XXX** |
| mAP@50 | X.XXX |
| mAP@75 | X.XXX |
| mAP (small) | X.XXX |
| mAP (medium) | X.XXX |
| mAP (large) | X.XXX |

### Per-class (IoU=0.50)

| Class | Precision | Recall | F1 | GT |
|---|---|---|---|---|
| dent | ... |
| scratch | ... |
| crack | ... |
| glass shatter | ... |
| lamp broken | ... |
| tire flat | ... |

### Latency (RTX 4070 Laptop, N runs)

| | ms |
|---|---|
| Mean | X.X |
| Std | X.X |
| p50 | X.X |
| p95 | X.X |
| p99 | X.X |

---

## Comparison vs Previous Damage Model

The previous damage model was trained on the "Car parts dataset" (Supervisely format, 814 images, 8 classes) on a Jetson Orin Nano.

| Metric | Old (Jetson, 814 img, 8 cls) | CarDD (RTX 4070, 2816 img, 6 cls) | Delta |
|---|---|---|---|
| mAP (0.50:0.95) | 0.040 | X.XXX | +X.XXX |
| mAP@50 | 0.082 | X.XXX | +X.XXX |
| mAP@75 | 0.035 | X.XXX | +X.XXX |

### Key differences

- **Dataset size**: 2816 vs 814 training images (3.5× more)
- **Annotation quality**: CarDD has clean COCO polygon masks; old dataset had high visual-similarity classes (Dent/Cracked/Broken) and severe imbalance (Scratch 592 GT vs Cracked 21 GT)
- **Classes**: CarDD uses 6 distinct damage types with clearer visual boundaries (lamp broken, tire flat are unambiguous); old set had 8 types with significant overlap
- **Hardware**: RTX 4070 allowed batch=4 and full 800/1333 px resolution vs batch=1, 640/800 px on Jetson
```

Fill in actual values from `models/cardd_eval_results.json`.

- [ ] **Step 2: Commit everything**

```bash
git add backend/mask_rcnn/config.py \
        backend/mask_rcnn/dataset_coco.py \
        backend/mask_rcnn/model.py \
        backend/mask_rcnn/train.py \
        backend/mask_rcnn/evaluate.py \
        run_maskrcnn.py \
        CARDD_TRAINING_SUMMARY.md \
        models/cardd_model.history.json \
        models/cardd_eval_results.json
git commit -m "feat: CarDD Mask R-CNN training and evaluation results"
```

(Do not commit `models/cardd_model.pth` — it's ~169 MB and likely in .gitignore already.)
