---
title: Auto-Owners Vehicle Damage Estimator
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Auto-Owners Vehicle Damage Estimator

## Description

This project automates vehicle damage assessment for Auto-Owners Insurance using a multi-model computer vision and generative AI pipeline. A user uploads a photo of a damaged vehicle and the system performs instance segmentation to identify which parts are damaged, classifies the damage type, and generates a structured output for downstream cost estimation. The pipeline is designed to be transparent — returning part-level detections with confidence scores, bounding boxes, segmentation masks, and severity ratings rather than a single opaque result.

## Models

| Model | Purpose |
|---|---|
| **Mask R-CNN** (ResNet-50-FPN) | Instance segmentation for car part detection (22 classes) |
| **YOLOv8m** | Object detection for damage type classification (6 classes) |
| **Gemini** *(planned)* | Multimodal LLM for natural language damage explanation and cost reasoning |

The Mask R-CNN model was fine-tuned from a COCO-pretrained ResNet-50-FPN backbone using a two-phase transfer learning strategy: Phase 1 freezes the backbone and trains only the RPN and ROI heads; Phase 2 unfreezes all layers for full fine-tuning. Training used SGD with momentum, StepLR scheduling, mixed precision (AMP autocast + GradScaler), and gradient checkpointing to fit within 8 GB unified memory on a Jetson Orin Nano (CUDA 12.6, JetPack 6.1). The YOLOv8m damage model was fine-tuned separately on a labeled vehicle damage dataset.

## Computer Vision Pipeline

```
Image Upload
    │
    ▼
Preprocessing (orientation correction, quality gate: blur detection, brightness check)
    │
    ├──▶ Parts Mask R-CNN ──▶ Part detections (class, bbox, mask, score)
    │
    ├──▶ Damage YOLOv8m ──▶ Damage detections (class, bbox, score)
    │
    ▼
IoU / Mask Overlap Cross-Reference
    │
    ▼
Structured JSON Output (part, damage type, severity, confidence, bbox)
    │
    ▼
LLM Layer — Gemini (planned): cost reasoning + natural language explanation
```

## Installation & Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Model weights: `parts_model.pth` and `best_car_damage_yolo.pt` (place in `/models` at project root)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart pillow numpy \
    opencv-python-headless pycocotools ultralytics torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
uvicorn api:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

The application will be available at `http://localhost:3000`

### For Production

Backend is deployed on **Hugging Face Spaces** (Docker):
```
https://eerabhatt-ao-damage-estimation.hf.space
```

Frontend is deployed on **Vercel**:
```
https://ao-damage-estimation.vercel.app
```

Both auto-redeploy on every push to `main`.

## Project Structure

```
ao-damage-estimation/
├── backend/
│   ├── api.py              # FastAPI REST API — POST /detect endpoint
│   ├── cv_detector.py      # Computer vision wrapper around Mask R-CNN inference pipeline
│   └── mask_rcnn/
│       ├── config.py       # Hyperparameters, class labels, paths
│       ├── model.py        # Mask R-CNN model factory (FastRCNNPredictor, MaskRCNNPredictor)
│       ├── dataset.py      # Custom PyTorch Dataset with COCO-format annotations
│       ├── train.py        # Two-phase training loop with AMP and gradient checkpointing
│       ├── inference.py    # Full inference pipeline with mask overlap cross-referencing
│       ├── evaluate.py     # COCO mAP evaluation (pycocotools)
│       └── preprocess.py   # Image quality gating and orientation correction
├── frontend/
│   └── src/                # React SPA
├── models/                 # Model weights (not in git — hosted on HF Hub)
└── Dockerfile              # Containerized deployment for HF Spaces
```

## Model Weights

Weights are not stored in git. They are hosted on Hugging Face Hub at `eerabhatt/ao-damage-models` and pulled automatically at Docker build time via `huggingface_hub`.

To run locally, download `parts_model.pth` and `best_car_damage_yolo.pt` from the HF Hub and place them in `/models`.

## Model Performance

| Model | mAP@50 | Notes |
|---|---|---|
| Parts Mask R-CNN | 0.785 | Strong on large parts (door, windshield, wheel) |
| Damage YOLOv8m | 0.751 | 6-class damage detection (dent, scratch, crack, glass shatter, lamp broken, tire flat) |

## API

### `POST /detect`

Upload a vehicle photo and receive structured damage detections.

**Request:** `multipart/form-data` with `image` field

**Response:**
```json
{
  "detections": [
    {
      "part": "Front-bumper",
      "damage_type": "Dent",
      "confidence": 0.73,
      "severity": "moderate",
      "bbox": [120, 340, 450, 520],
      "iou": 0.42
    }
  ],
  "summary": {
    "total_damaged_parts": 2,
    "parts": ["Front-bumper", "Hood"],
    "damage_types": ["Dent", "Scratch"]
  },
  "inference_ms": 842.3
}
```

### `GET /health`

Returns model status.

## Technologies Used

**Frontend:** React, Tailwind CSS, Framer Motion  
**Backend:** FastAPI, Uvicorn  
**Computer Vision:** PyTorch, Mask R-CNN (ResNet-50-FPN), YOLOv8m (Ultralytics), OpenCV, NumPy, Pillow, torchvision, pycocotools  
**LLM (planned):** Gemini, Google Generative AI SDK  
**Deployment:** Docker, Hugging Face Spaces, Vercel

## License

This project is licensed under the MIT License.
