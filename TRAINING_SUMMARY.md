# Mask R-CNN Training Summary

## Platform

| | |
|---|---|
| Device | Jetson Orin Nano |
| JetPack | 6.1 (L4T R36.4.7) |
| CUDA | 12.6 |
| Python | 3.10.12 |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 (NVIDIA ARM64 wheel) |
| torchvision | 0.20.0 (built from source against NVIDIA torch) |

### Jetson-specific setup notes

- PyTorch installed from `https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/` (direct wheel URL — the index does not support PEP 503)
- `libcusparseLt.so.0` is not bundled with JetPack 6.1; installed via `pip install nvidia-cusparselt-cu12` and path baked into `.venv/bin/activate`
- torchvision PyPI aarch64 wheels are compiled against stock PyTorch and will fail at import (`torchvision::nms` operator mismatch); must build from source with `--no-build-isolation`
- Training config: `BATCH_SIZE=1`, `NUM_WORKERS=2`, gradient checkpointing on all ResNet-50 Bottleneck blocks, image resize capped at 640/800px — required to stay within 8 GB unified memory during phase 2 full fine-tune

---

## Training Configuration

| | |
|---|---|
| Architecture | Mask R-CNN, ResNet-50-FPN backbone (COCO pretrained) |
| Phase 1 | 10 epochs, backbone frozen, lr=0.005 |
| Phase 2 | 30 epochs, full fine-tune, lr=0.001 (CosineAnnealingLR) |
| Optimizer | SGD, momentum=0.9, weight_decay=5e-4 |
| Mixed precision | AMP (autocast + GradScaler) |

---

## Parts Model (`models/parts_model.pth`)

21 classes: Back-bumper, Back-door, Back-wheel, Back-window, Back-windshield, Fender, Front-bumper, Front-door, Front-wheel, Front-window, Grille, Headlight, Hood, License-plate, Mirror, Quarter-panel, Rocker-panel, Roof, Tail-light, Trunk, Windshield.

### Metrics

| Metric | Value |
|---|---|
| mAP (COCO) | **0.507** |
| mAP@50 | 0.785 |
| mAP@75 | 0.559 |
| mAP (small) | 0.088 |
| mAP (medium) | 0.367 |
| mAP (large) | 0.565 |

### Latency (30 runs, Jetson Orin)

| | ms |
|---|---|
| Mean | 412.1 |
| Std | 26.7 |
| p50 | 416.3 |
| p95 | 444.2 |
| p99 | 459.7 |

### Per-class F1 highlights

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Front-door | 0.947 | 0.957 | 0.952 |
| Windshield | 0.918 | 0.924 | 0.921 |
| Front-wheel | 0.893 | 0.961 | 0.926 |
| Hood | 0.895 | 0.906 | 0.901 |
| Back-door | 0.875 | 0.907 | 0.891 |
| Trunk | 0.397 | 0.530 | 0.454 |
| Back-bumper | 0.684 | 0.713 | 0.698 |

---

## Damage Model (`models/damage_model.pth`)

8 classes: Broken part, Corrosion, Cracked, Dent, Flaking, Missing part, Paint chip, Scratch.

### Metrics

| Metric | Value |
|---|---|
| mAP (COCO) | **0.040** |
| mAP@50 | 0.082 |
| mAP@75 | 0.035 |
| mAP (small) | 0.015 |
| mAP (medium) | 0.037 |
| mAP (large) | 0.068 |

### Latency (30 runs, Jetson Orin)

| | ms |
|---|---|
| Mean | 384.1 |
| Std | 45.3 |
| p50 | 399.0 |
| p95 | 439.2 |
| p99 | 444.8 |

### Per-class F1

| Class | Precision | Recall | F1 | GT count |
|---|---|---|---|---|
| Missing part | 0.374 | 0.474 | 0.418 | 154 |
| Broken part | 0.118 | 0.308 | 0.171 | 299 |
| Dent | 0.100 | 0.195 | 0.132 | 375 |
| Scratch | 0.011 | 0.034 | 0.016 | 592 |
| Cracked | 0.050 | 0.048 | 0.049 | 21 |
| Paint chip | 0.005 | 0.014 | 0.008 | 370 |
| Corrosion | 0.000 | 0.000 | 0.000 | 51 |
| Flaking | 0.000 | 0.000 | 0.000 | 44 |

### Why damage mAP is low

The damage model's mAP of 0.04 reflects fundamental dataset challenges rather than a training failure:

- **Severe class imbalance**: Scratch (592 GT) vs Cracked (21 GT) — a 28x ratio
- **High visual similarity**: Dent, Cracked, and Broken part overlap significantly in appearance
- **Small damage regions**: Most damage annotations are small objects, where Mask R-CNN struggles (mAP_small=0.015)
- **Dataset size**: ~800–1000 images split across 8 classes is insufficient for fine-grained damage localisation

### Suggested improvements

- Collect 3–5x more annotated damage images, prioritising rare classes (Corrosion, Flaking, Cracked)
- Apply class-weighted sampling or loss weighting to address imbalance
- Use pseudo-labelling on unannotated images to expand the training set
- Consider a Swin-T or ResNet-101 backbone for better small-object detection
- Mosaic or CutMix augmentation to synthetically increase effective dataset size

---

## File inventory

| File | Description |
|---|---|
| `models/parts_model.pth` | Parts model weights (~169 MB) |
| `models/damage_model.pth` | Damage model weights (~169 MB) |
| `models/parts_model.history.json` | Per-epoch loss history, parts run |
| `models/damage_model.history.json` | Per-epoch loss history, damage run |
| `models/parts_eval_results.json` | Full parts evaluation results |
| `models/damage_eval_results.json` | Full damage evaluation results |
| `run_maskrcnn.py` | CLI entry point (train / eval / infer / run) |
| `backend/mask_rcnn/` | Pipeline source (config, dataset, model, train, evaluate, inference) |
