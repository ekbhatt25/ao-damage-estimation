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

PARTS_MODEL_PATH  = MODELS_DIR / "parts_model.pth"
DAMAGE_MODEL_PATH = MODELS_DIR / "damage_model.pth"

CARDD_DATA_DIR   = DATA_ROOT / "CarDD_release" / "CarDD_COCO"
CARDD_TRAIN_DIR  = CARDD_DATA_DIR / "train2017"
CARDD_VAL_DIR    = CARDD_DATA_DIR / "val2017"
CARDD_TRAIN_ANN  = CARDD_DATA_DIR / "annotations" / "instances_train2017.json"
CARDD_VAL_ANN    = CARDD_DATA_DIR / "annotations" / "instances_val2017.json"
CARDD_MODEL_PATH = MODELS_DIR / "cardd_model.pth"

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
    "Broken part", "Corrosion", "Cracked", "Dent",
    "Flaking", "Missing part", "Paint chip", "Scratch",
]

NUM_PART_CLASSES   = len(PART_CLASSES)    # 22
NUM_DAMAGE_CLASSES = len(DAMAGE_CLASSES)  # 9

PART_LABEL_MAP   = {c: i for i, c in enumerate(PART_CLASSES)}
DAMAGE_LABEL_MAP = {c: i for i, c in enumerate(DAMAGE_CLASSES)}

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
SCORE_THRESHOLD     = 0.50
NMS_IOU_THRESHOLD   = 0.50

# Damage-part overlap: minimum IoU to say a part is "affected"
PART_DAMAGE_OVERLAP_THRESHOLD = 0.10

# ── Latency benchmark ──────────────────────────────────────────────────────
LATENCY_WARMUP_RUNS = 5
LATENCY_BENCH_RUNS  = 30

# ── CarDD / RTX 4070 Laptop overrides ──────────────────────────────────────
CARDD_BATCH_SIZE    = 2    # Phase 2 full fine-tune at 800/1333px OOMs at batch=4 on 8 GB VRAM
CARDD_NUM_WORKERS   = 4
CARDD_PHASE1_EPOCHS = 10
CARDD_PHASE2_EPOCHS = 40   # more data warrants more fine-tune epochs
CARDD_PHASE1_LR     = 0.005
CARDD_PHASE2_LR     = 0.001
