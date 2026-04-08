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

This project automates vehicle damage assessment for Auto-Owners Insurance using a multi-model computer vision, machine learning, and generative AI pipeline. A user uploads a photo of a damaged vehicle and the system identifies which parts are damaged, classifies the damage type, estimates repair costs, and produces a straight-through processing (STP) eligibility decision. The pipeline returns part-level detections with confidence scores, bounding boxes, segmentation masks, severity ratings, per-part cost ranges, and a Gemini-generated natural language explanation.

## Models

| Model | Purpose |
|---|---|
| **Mask R-CNN** (ResNet-50-FPN) | Instance segmentation for car part detection (22 classes) |
| **YOLOv8m** | Object detection for damage type classification (6 classes) |
| **GradientBoosting Regressor** | ML cost estimation — predicts repair cost from part, damage type, and severity |
| **Gemini 1.5 Flash** | LLM for natural language damage explanation and STP reasoning |

The Mask R-CNN model was fine-tuned from a COCO-pretrained ResNet-50-FPN backbone using a two-phase transfer learning strategy: Phase 1 freezes the backbone and trains only the RPN and ROI heads; Phase 2 unfreezes all layers for full fine-tuning. Training used SGD with momentum, StepLR scheduling, mixed precision (AMP autocast + GradScaler), and gradient checkpointing to fit within 8 GB unified memory on a Jetson Orin Nano (CUDA 12.6, JetPack 6.1). The YOLOv8m damage model was fine-tuned separately on a labeled vehicle damage dataset. The cost estimation model is a scikit-learn GradientBoostingRegressor trained on repair cost data sourced from RepairPal and cross-referenced against SCRS/ASA labor rate surveys.

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
IoU / Mask Overlap Cross-Reference + Severity Assignment
    │
    ▼
Cost Estimation (GradientBoosting ML model)
    ├── Repair cost per part (sourced from RepairPal)
    ├── Regional labor rate adjustment by ZIP (body / mechanical / paint)
    │   sourced from SCRS, ASA, and state Departments of Insurance
    ├── Total cost range
    └── Total loss flag (repair cost > 70% of estimated ACV)
    │
    ▼
Gemini 1.5 Flash — natural language explanation + STP eligibility decision
    └── STP criteria: cost < $1,500, confidence > 80%, no major damage, not a total loss
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
    opencv-python-headless pycocotools ultralytics scikit-learn joblib \
    google-generativeai python-dotenv torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
uvicorn api:app --reload --port 8000
```

Set the following environment variables (or create a `.env` file in `/backend`):
```
GEMINI_API_KEY=your_key_here
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
│   ├── cv_detector.py      # CV wrapper: runs Mask R-CNN + YOLO, cross-references results
│   ├── cost_estimator.py   # ML cost estimation (GradientBoosting) with labor rate adjustment
│   ├── llm_client.py       # Gemini integration: explanation generation + STP decision
│   ├── data/
│   │   ├── repair_costs.csv    # Part repair/replace costs (sourced from RepairPal)
│   │   └── labor_rates.csv     # Body/mechanical/paint rates by state (SCRS, ASA, DoIs)
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

Upload a vehicle photo and receive structured damage detections, cost estimates, and STP decision.

**Request:** `multipart/form-data` with `image` and `zipCode` fields

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
  "cost": {
    "damaged_parts": [
      {
        "part": "Front-bumper",
        "damage_type": "Dent",
        "severity": "moderate",
        "action": "repair",
        "labor_category": "body",
        "labor_rate": 58.0,
        "cost_range": [373, 505]
      }
    ],
    "total_cost_range": [373, 505],
    "zip_code": "48823",
    "labor_rates": {"body": 58.0, "mechanical": 80.0, "paint": 52.0},
    "acv_estimate": 20000,
    "total_loss": false
  },
  "explanation": "Your vehicle sustained a moderate dent to the front bumper...",
  "confidence_score": 0.84,
  "stp_eligible": true,
  "stp_reasoning": "Claim eligible for auto-approval: cost $439 under $1,500 threshold...",
  "inference_ms": 842.3
}
```

### `GET /health`

Returns model load status.

## Technologies Used

**Frontend:** React, Tailwind CSS, Framer Motion  
**Backend:** FastAPI, Uvicorn  
**Computer Vision:** PyTorch, Mask R-CNN (ResNet-50-FPN), YOLOv8m (Ultralytics), OpenCV, NumPy, Pillow, torchvision, pycocotools  
**Cost Estimation:** scikit-learn (GradientBoostingRegressor), joblib  
**LLM:** Gemini 1.5 Flash, Google Generative AI SDK  
**Deployment:** Docker, Hugging Face Spaces, Vercel

## License

This project is licensed under the MIT License.
