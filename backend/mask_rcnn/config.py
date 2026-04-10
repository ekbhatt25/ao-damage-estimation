"""
Central configuration for the Mask R-CNN car damage pipeline.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
DATA_ROOT = ROOT / "data"

# Note: folder names in the zip are swapped vs what you'd expect
PARTS_IMG_DIR = DATA_ROOT / "Car damages dataset" / "File1" / "img"
PARTS_ANN_DIR = DATA_ROOT / "Car damages dataset" / "File1" / "ann"

DAMAGE_IMG_DIR = DATA_ROOT / "Car parts dataset" / "File1" / "img"
DAMAGE_ANN_DIR = DATA_ROOT / "Car parts dataset" / "File1" / "ann"

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

PARTS_MODEL_PATH       = MODELS_DIR / "parts_model.pth"
DAMAGE_MODEL_PATH      = MODELS_DIR / "damage_model.pth"
YOLO_DAMAGE_MODEL_PATH = MODELS_DIR / "best_car_damage_yolo.pt"
SEVERITY_MODEL_PATH    = MODELS_DIR / "severity_yolov8_cls.pt"
SEVERITY_MODEL_HF_REPO = "nezahatkorkmaz/car-damage-level-detection-yolov8"
SEVERITY_MODEL_HF_FILE = "car-damage.pt"

# ── Classes ────────────────────────────────────────────────────────────────
# Index 0 is always background for Mask R-CNN
PART_CLASSES = [
    "__background__",
    "Back-bumper", "Back-door", "Back-wheel", "Back-window", "Back-windshield",
    "Fender", "Front-bumper", "Front-door", "Front-wheel", "Front-window",
    "Grille", "Headlight", "Hood", "License-plate", "Mirror",
    "Quarter-panel", "Rocker-panel", "Roof", "Tail-light", "Trunk", "Windshield",
]

DAMAGE_CLASSES = [
    "__background__",
    "Dent", "Scratch", "Crack", "Glass shatter", "Lamp broken", "Tire flat",
]

NUM_PART_CLASSES   = len(PART_CLASSES)    # 22
NUM_DAMAGE_CLASSES = len(DAMAGE_CLASSES)  # 7

PART_LABEL_MAP   = {c: i for i, c in enumerate(PART_CLASSES)}
DAMAGE_LABEL_MAP = {c: i for i, c in enumerate(DAMAGE_CLASSES)}

# ── Preprocessing ──────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Quality gate thresholds
MIN_RESOLUTION      = 100     # minimum side length in px
BLUR_THRESHOLD      = 80.0    # Laplacian variance below this → blurry
BRIGHTNESS_MIN      = 20.0    # mean pixel value (0-255)
BRIGHTNESS_MAX      = 235.0

# ── Training ───────────────────────────────────────────────────────────────
TRAIN_VAL_SPLIT     = 0.80
RANDOM_SEED         = 42
BATCH_SIZE          = 1       # reduced for Jetson Orin Nano 8 GB unified memory
NUM_WORKERS         = 2

# Phase 1: train only the prediction heads (backbone frozen)
PHASE1_EPOCHS       = 10
PHASE1_LR           = 0.005

# Phase 2: fine-tune everything
PHASE2_EPOCHS       = 30
PHASE2_LR           = 0.001

LR_MOMENTUM         = 0.9
LR_WEIGHT_DECAY     = 5e-4
LR_STEP_SIZE        = 10     # StepLR step size (epochs)
LR_GAMMA            = 0.5

# Detection score threshold for inference
SCORE_THRESHOLD     = 0.30
NMS_IOU_THRESHOLD   = 0.50

# Damage-part overlap: minimum IoU to say a part is "affected"
PART_DAMAGE_OVERLAP_THRESHOLD = 0.05

# ── Latency benchmark ──────────────────────────────────────────────────────
LATENCY_WARMUP_RUNS = 5
LATENCY_BENCH_RUNS  = 30
