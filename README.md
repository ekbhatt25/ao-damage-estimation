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

This project automates vehicle damage assessment for Auto-Owners Insurance using a multi-model computer vision, machine learning, and generative AI pipeline. A user selects their state, uploads a photo of a damaged vehicle, and the system identifies which parts are damaged, classifies the damage type and severity, estimates repair costs using regional labor rates, and produces a straight-through processing (STP) eligibility decision.

The frontend renders the uploaded image with color-coded bounding box overlays (yellow = minor, orange = moderate, red = severe) alongside part-level detections with confidence scores, severity ratings, per-part cost ranges, a natural language explanation, and fraud signal flags. After analysis, adjusters can change the state to instantly re-price all parts using that region's labor rates, or manually override individual part detections and costs.

Session-based claim history is stored in an append-only audit log and accessible via a slide-in sidebar with CSV export.

## Models

| Model | Purpose |
|---|---|
| **Mask R-CNN** (ResNet-50-FPN) | Instance segmentation for car part detection (22 classes, mAP@50 = 0.785) |
| **YOLOv8m** | Object detection for damage type classification (6 classes, mAP@50 = 0.751) |
| **YOLOv8n-cls** | Image classification for damage severity (minor / moderate / severe) |
| **GradientBoosting Regressor** | ML cost estimation — predicts repair cost from part, damage type, and severity |
| **Gemini 2.5 Flash** | LLM for natural language damage explanation and STP reasoning (rule-based fallback if API key unavailable) |

Mask R-CNN was fine-tuned using two-phase transfer learning (frozen backbone → full fine-tuning) with AMP and gradient checkpointing. YOLOv8m was fine-tuned on a labeled vehicle damage dataset. The severity classifier is sourced from `nezahatkorkmaz/car-damage-level-detection-yolov8`. The cost model is trained on repair cost estimates scaled by SCRS 2024 labor rate survey data (body $67/hr, paint $65/hr national medians; mechanical $95/hr).

## Computer Vision Pipeline

```
Image Upload
    │
    ▼
Preprocessing
    ├── Resolution cap (800px max — CPU inference optimization)
    ├── Orientation correction (EXIF)
    ├── Quality gate (blur, brightness, resolution)
    └── Fraud signals — flagged in API response and audit log
            ├── low_pixel_variance: nearly uniform image — suggests a solid fill
            │   or digitally generated image rather than a real damage photo
            ├── editing_software_detected: EXIF Software tag contains Photoshop,
            │   GIMP, Lightroom, etc. — image was manipulated after capture
            └── duplicate_image: perceptual hash matches a photo submitted within
                the last 60 seconds — same damage being claimed more than once
    │
    ├──▶ Mask R-CNN (parts) ──▶ Part detections (class, bbox, mask, score)
    │
    ├──▶ YOLOv8m (damage) ──▶ Damage detections (class, bbox, score)
    │
    ▼
IoU / Mask Overlap Cross-Reference
    └── YOLOv8n-cls (severity) ──▶ Minor / Moderate / Severe per damage crop
    │
    ▼
Cost Estimation (GradientBoosting ML model)
    ├── Repair cost per part
    ├── Regional labor rates by state — body / mechanical / paint (SCRS 2024 survey data)
    ├── Total cost range (±15% band)
    └── Total loss flag (repair cost > 70% of estimated ACV)
    │
    ▼
Gemini Flash — natural language explanation + STP eligibility decision
    ├── STP criteria: cost < $1,500, confidence > 60%, not a total loss
    └── Auto-escalation to adjuster if confidence < 40% or total loss
    │
    ▼
Audit Trail (JSONL) — claim ID, session ID, timestamp, model version, full decision log
```

## Key Features

- **State-based labor rates** — select a state before upload to apply SCRS 2024 regional rates; body rates range from ~$59/hr (Southeast) to ~$84/hr (West Coast)
- **Live cost re-estimation** — change the state dropdown in the results panel to instantly re-price all detected parts without re-uploading
- **Adjuster overrides** — edit any part's detection (part, damage type, severity) and get a backend-recalculated cost range; override takes priority over state adjustments
- **Fraud signals** — three passive checks (pixel variance, EXIF editing software, duplicate hash) flagged on every submission
- **Session claim history** — each browser session has a unique ID; all claims for the session are viewable in a slide-in sidebar and exportable as CSV
- **Append-only audit log** — every claim is logged with full model inputs/outputs for compliance review

## Installation & Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for containerized local run or production build)
- Model weights hosted on Hugging Face Hub at `eerabhatt/ao-damage-models` — pulled automatically at build time

### Option A — Run with Docker (recommended)

Install Docker: [docs.docker.com/get-docker](https://docs.docker.com/get-docker)

```bash
docker build -t ao-damage-estimation .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key_here ao-damage-estimation
```

The API will be available at `http://localhost:7860`

### Option B — Run without Docker

**Backend:**

```bash
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

### Production Deployment

Backend — **Hugging Face Spaces** (Docker): `https://eerabhatt-ao-damage-estimation.hf.space`

Frontend — **Vercel**: `https://ao-damage-estimation.vercel.app`

Frontend auto-redeploys on every push to `main`. Backend requires a separate push to the HF Spaces git remote:

```bash
git push space main
```

## Project Structure

```
ao-damage-estimation/
├── backend/
│   ├── api.py              # FastAPI REST API — /detect, /estimate, /claims endpoints
│   ├── cv_detector.py      # CV wrapper: runs Mask R-CNN + YOLO, cross-references results
│   ├── cost_estimator.py   # ML cost estimation (GradientBoosting) with state labor rate adjustment
│   ├── llm_client.py       # Gemini integration: explanation generation + STP decision
│   ├── audit_logger.py     # JSONL audit trail — one record per claim
│   ├── fraud_detector.py   # Perceptual hash duplicate detection + EXIF metadata anomaly detection
│   ├── data/
│   │   ├── repair_costs.csv    # Part repair/replace costs
│   │   └── labor_rates.csv     # Body/mechanical/paint rates by state (SCRS 2024)
│   └── mask_rcnn/
│       ├── config.py       # Hyperparameters, class labels, paths
│       ├── model.py        # Mask R-CNN model factory
│       ├── dataset.py      # Custom PyTorch Dataset with COCO-format annotations
│       ├── train.py        # Two-phase training loop with AMP and gradient checkpointing
│       ├── inference.py    # Full inference pipeline with mask overlap cross-referencing
│       ├── evaluate.py     # COCO mAP evaluation
│       └── preprocess.py   # Image quality gating, orientation correction, fraud signals
├── frontend/
│   └── src/
│       └── components/
│           ├── ImageOverlay.js     # HTML5 Canvas bbox overlay, color-coded by severity
│           ├── ResultsDisplay.js   # Full results panel — STP, cost, state selector, overrides, history
│           ├── ImageUpload.js      # Drag-and-drop image uploader
│           └── LoadingOverlay.js   # Analysis loading state with cancel button
├── download_models.py      # Downloads model weights from HF Hub at build time
├── models/                 # Model weights (not in git — hosted on HF Hub)
├── audit_log.jsonl         # Append-only claim audit log (not in git)
└── Dockerfile              # Containerized deployment for HF Spaces
```

## Model Weights

Weights are hosted on Hugging Face Hub at `eerabhatt/ao-damage-models` and pulled automatically at Docker build time via `download_models.py`. To run locally, download the following and place in `/models`:

- `parts_model.pth` — Mask R-CNN parts detector
- `best_car_damage_yolo.pt` — YOLOv8m damage detector
- `severity_yolov8_cls.pt` — YOLOv8n-cls severity classifier

## API

### `POST /detect`

Upload a vehicle photo and receive structured damage detections, cost estimates, and STP decision.

**Request:** `multipart/form-data` with `image`, `state` (2-letter abbreviation, optional), and `session_id` fields

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
        "labor_rate": 67.0,
        "cost_range": [373, 505]
      }
    ],
    "total_cost_range": [373, 505],
    "state": "MI",
    "labor_rates": {"body": 67.0, "mechanical": 95.0, "paint": 65.0},
    "acv_estimate": 20000,
    "total_loss": false
  },
  "explanation": "Your vehicle sustained a moderate dent to the front bumper...",
  "confidence_score": 0.81,
  "stp_eligible": true,
  "stp_reasoning": "Claim eligible for auto-approval: cost $439 under $1,500 threshold, 81% confidence meets requirement, not a total loss.",
  "requires_adjuster_review": false,
  "override_allowed": true,
  "model_version": "1.0.0",
  "fraud_flags": [],
  "claim_id": "d4d22393-32b5-4720-afdd-44977b980943",
  "state": "MI",
  "inference_ms": 842.3
}
```

### `GET /estimate`

Returns a cost estimate for a single part. Used by the frontend adjuster override and live state re-pricing.

**Query params:** `part`, `damage_type`, `severity`, `state` (optional)

### `GET /claims`

Returns claim history for a session from the audit log.

**Query params:** `session_id`, `limit` (default 50)

### `GET /health`

Returns model load status and LLM availability.

## Technologies Used

**Frontend:** React, Tailwind CSS, Framer Motion, HTML5 Canvas (bounding box overlay), Lucide React  
**Backend:** FastAPI, Uvicorn  
**Computer Vision:** PyTorch, Mask R-CNN (ResNet-50-FPN), YOLOv8m, YOLOv8n-cls (Ultralytics), OpenCV, NumPy, Pillow, torchvision, pycocotools  
**Cost Estimation:** scikit-learn (GradientBoostingRegressor), joblib, SCRS 2024 labor rate data  
**LLM:** Gemini 2.5 Flash, Google GenAI SDK (`google-genai`)  
**Deployment:** Docker, Hugging Face Spaces, Vercel, Hugging Face Hub

## License

This project is licensed under the MIT License.
