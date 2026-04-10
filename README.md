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

Mask R-CNN was fine-tuned using two-phase transfer learning (frozen backbone → full fine-tuning) with AMP and gradient checkpointing on a Jetson Orin Nano. YOLOv8m was fine-tuned on a labeled vehicle damage dataset. The cost model was trained on repair cost data sourced from RepairPal, cross-referenced against SCRS/ASA labor rate surveys.

## Computer Vision Pipeline

```
Image Upload
    │
    ▼
Preprocessing
    ├── Orientation correction (EXIF)
    ├── Quality gate (blur, brightness, resolution)
    └── Fraud signals (pixel variance, sharpness anomaly, aspect ratio)
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
    ├── Regional labor rates by ZIP — body / mechanical / paint
    │   sourced from SCRS, ASA, and state Departments of Insurance
    ├── Total cost range (±15% band)
    └── Total loss flag (repair cost > 70% of estimated ACV)
    │
    ▼
Gemini 1.5 Flash — natural language explanation + STP eligibility decision
    ├── STP criteria: cost < $1,500, confidence > 80%, no major damage, not a total loss
    └── Auto-escalation to adjuster if confidence < 60% or total loss
    │
    ▼
Audit Trail (JSONL) — claim ID, timestamp, model version, full decision log
```

## Installation & Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for containerized local run or production build)
- Model weights: `parts_model.pth` and `best_car_damage_yolo.pt` — download from `eerabhatt/ao-damage-models` on Hugging Face and place in `/models` at project root

### Option A — Run with Docker (recommended)

```bash
# Build and run the backend locally
docker build -t ao-damage-estimation .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key_here ao-damage-estimation
```

The API will be available at `http://localhost:7860`

Install Docker: [docs.docker.com/get-docker](https://docs.docker.com/get-docker)

### Option B — Run without Docker

**Backend:**

```bash
# Install system dependency required by OpenCV
# macOS:
brew install libgl1
# Ubuntu/Debian:
sudo apt-get install libgl1

cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install fastapi "uvicorn[standard]" python-multipart pillow numpy \
    opencv-python-headless pycocotools ultralytics scikit-learn joblib \
    google-generativeai python-dotenv torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
uvicorn api:app --reload --port 8000
```

Create a `.env` file in `/backend`:
```
GEMINI_API_KEY=your_key_here
```

**Frontend:**

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
│   ├── audit_logger.py     # JSONL audit trail — one record per claim
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
│       └── preprocess.py   # Image quality gating, orientation correction, fraud signals
├── frontend/
│   └── src/                # React SPA
├── models/                 # Model weights (not in git — hosted on HF Hub)
├── audit_log.jsonl         # Append-only claim audit log (not in git)
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
  "requires_adjuster_review": false,
  "override_allowed": true,
  "model_version": "1.0.0",
  "fraud_flags": [],
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
