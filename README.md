---
title: AO Damage Estimation
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AO Vehicle Damage Estimator

## Description

This project automates vehicle damage assessment for Auto-Owners Insurance using a multi-model AI pipeline. A user uploads a photo of a damaged vehicle and the system identifies which parts are damaged, what type of damage is present, and generates a structured output for downstream cost estimation. The pipeline is designed to be transparent — returning part-level detections with confidence scores and severity ratings rather than a single opaque result.

## Models

| Model | Purpose |
|---|---|
| **Mask R-CNN** (ResNet-50-FPN) | Detects damaged car parts (22 classes: bumper, hood, door, etc.) |
| **Mask R-CNN** (ResNet-50-FPN) | Detects damage types (8 classes: dent, scratch, broken part, etc.) |
| **Gemini** *(planned)* | Generates natural language explanation of damage and repair recommendations |
| **YOLOv8** *(planned)* | Alternative/supplementary damage type detection |

Both Mask R-CNN models were trained from scratch on the Car Parts and Car Damages datasets using a two-phase training strategy (frozen backbone → full fine-tune) on a Jetson Orin Nano.

## Installation & Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Model weights: `parts_model.pth` and `damage_model.pth` (place in `/models` at project root)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart pillow numpy \
    opencv-python-headless pycocotools torch torchvision \
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
│   ├── api.py              # FastAPI app — POST /detect endpoint
│   ├── cv_detector.py      # Wrapper around Mask R-CNN inference pipeline
│   └── mask_rcnn/          # Mask R-CNN model, training, inference, config
├── frontend/
│   └── src/                # React app
├── models/                 # Model weights (not in git — see below)
└── Dockerfile              # HF Spaces deployment
```

## Model Weights

Weights are not stored in git (~170 MB each). They are hosted on Hugging Face Hub at `eerabhatt/ao-damage-models` and downloaded automatically at deployment build time.

To run locally, download `parts_model.pth` and `damage_model.pth` from the HF Hub and place them in `/models`.

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
      "bbox": [120, 340, 450, 520]
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
**Backend:** FastAPI, PyTorch, torchvision  
**Models:** Mask R-CNN (ResNet-50-FPN), YOLOv8 *(planned)*, Gemini *(planned)*  
**Deployment:** Hugging Face Spaces (Docker), Vercel

## License

This project is licensed under the MIT License.
