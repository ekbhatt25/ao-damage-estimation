# CarDD Mask R-CNN Training Summary

## Platform

| | |
|---|---|
| Device | ASUS ROG Flow X16 (2023) — RTX 4070 Laptop GPU |
| VRAM | 8.2 GB |
| CUDA | 12.4 |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |

### Hardware notes

- Phase 2 OOMed at batch=4 with 800/1333px resolution; reduced to batch=2
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set to reduce fragmentation
- Gradient checkpointing disabled (not needed on RTX 4070; only used on Jetson)

---

## Dataset — CarDD

| | |
|---|---|
| Source | CarDD (COCO format, official train/val/test split) |
| Train images | 2816 |
| Val images | 810 |
| Train annotations | 6211 |
| Val annotations | 1744 |
| Classes | 6: dent, scratch, crack, glass shatter, lamp broken, tire flat |

---

## Training Configuration

| | |
|---|---|
| Architecture | Mask R-CNN ResNet-50-FPN (COCO pretrained) |
| Image size | 800–1333 px |
| Phase 1 | 10 epochs, backbone frozen, lr=0.005, StepLR |
| Phase 2 | 40 epochs, full fine-tune, lr=0.001, CosineAnnealingLR |
| Batch size | 2 |
| Optimizer | SGD, momentum=0.9, weight_decay=5e-4 |
| Mixed precision | AMP (autocast + GradScaler) |

---

## CarDD Model Results

### mAP (segmentation, val2017)

| Metric | Value |
|---|---|
| **mAP (0.50:0.95)** | **0.457** |
| mAP@50 | 0.617 |
| mAP@75 | 0.475 |
| mAP (small) | 0.046 |
| mAP (medium) | 0.054 |
| mAP (large) | 0.470 |

### Per-class (IoU=0.50)

| Class | Precision | Recall | F1 | GT |
|---|---|---|---|---|
| glass shatter | 0.970 | 0.978 | 0.974 | 135 |
| tire flat | 0.868 | 0.903 | 0.885 | 62 |
| lamp broken | 0.762 | 0.901 | 0.826 | 141 |
| scratch | 0.439 | 0.613 | 0.512 | 728 |
| dent | 0.439 | 0.603 | 0.508 | 501 |
| crack | 0.225 | 0.396 | 0.287 | 177 |

### Latency (RTX 4070 Laptop, 30 runs)

| | ms |
|---|---|
| Mean | 63.9 |
| Std | 4.9 |
| p50 | — |
| p95 | 72.9 |

---

## Comparison vs Previous Damage Model

The previous damage model was trained on the "Car parts dataset" (Supervisely format, 814 images, 8 classes) on a Jetson Orin Nano.

| Metric | Old (Jetson, 814 img, 8 cls) | CarDD (RTX 4070, 2816 img, 6 cls) | Delta |
|---|---|---|---|
| mAP (0.50:0.95) | 0.040 | **0.457** | +0.417 (+10×) |
| mAP@0.50 | 0.082 | **0.617** | +0.535 |
| mAP@0.75 | 0.035 | **0.475** | +0.440 |
| Mean latency | 384 ms | **64 ms** | −320 ms (6× faster) |

### Why the gap is so large

**Dataset quality** was the primary driver. The old dataset had severe class imbalance (Scratch: 592 GT vs Cracked: 21 GT — a 28× ratio), high visual similarity between classes (Dent/Cracked/Broken part), and 814 total images across 8 poorly-separated categories. CarDD's 6 classes have clear visual boundaries — glass shatter, lamp broken, and tire flat are largely unambiguous — which allows the model to learn strong discriminative features.

**Dataset size** (3.5× more images) and the use of official train/val splits contributed, but were secondary to annotation quality.

**Hardware** gave a 6× latency improvement (RTX 4070 vs Jetson Orin Nano) and allowed full 800/1333px resolution vs the Jetson-capped 640/800px.

### Remaining weaknesses

- **Crack detection is weak** (F1=0.287): crack annotations are often thin/elongated — hard for Mask R-CNN's anchor-based detector
- **Small object performance is poor** (mAP_small=0.046, mAP_medium=0.054): most damage regions are small relative to full-image size
- **Dent and scratch are mediocre** (F1 ~0.51): high visual similarity and fine-grained appearance variation

---

## File inventory

| File | Description |
|---|---|
| `models/cardd_model.pth` | Model weights (~169 MB) |
| `models/cardd_model.history.json` | Per-epoch loss history |
| `models/cardd_eval_results.json` | Full evaluation results |
| `models/cardd_train.log` | Full training log |
| `models/cardd_eval.log` | Full evaluation log |
| `backend/mask_rcnn/dataset_coco.py` | COCO-format dataset loader |
| `backend/mask_rcnn/config.py` | CarDD paths + hyperparams (CARDD_* constants) |
