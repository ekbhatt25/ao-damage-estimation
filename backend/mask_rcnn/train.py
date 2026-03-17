"""
Two-phase fine-tuning for Mask R-CNN.

Phase 1 (head warm-up):
  Backbone frozen. Only RPN + ROI box/mask heads trained.
  Prevents overwriting good pretrained features with noisy gradients early on.

Phase 2 (full fine-tune):
  All layers trainable. Lower LR to refine backbone features.

Usage:
  python -m backend.mask_rcnn.train --mode parts
  python -m backend.mask_rcnn.train --mode damage
"""
import argparse
import json
import time
from pathlib import Path
from typing import Literal

import torch
from torch.cuda.amp import GradScaler, autocast

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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=LR_MOMENTUM,
        weight_decay=LR_WEIGHT_DECAY,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> dict:
    model.train()
    total_loss = 0.0
    loss_components: dict[str, float] = {}
    n_batches = 0

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast():
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_loss = losses.item()
        total_loss += batch_loss
        n_batches  += 1
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v.item()

    avg = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    return {"total": avg, **avg_components}


def run_phase(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    n_epochs: int,
    phase_name: str,
    history: list,
) -> None:
    for epoch in range(1, n_epochs + 1):
        t0   = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        row = {"phase": phase_name, "epoch": epoch, **loss, "time_s": round(elapsed, 1)}
        history.append(row)

        comp_str = "  ".join(f"{k}={v:.4f}" for k, v in loss.items() if k != "total")
        print(
            f"[{phase_name}] epoch {epoch:3d}/{n_epochs}  "
            f"loss={loss['total']:.4f}  {comp_str}  ({elapsed:.1f}s)"
        )


def train(mode: Literal["parts", "damage", "cardd"]) -> Path:
    """
    Full two-phase training run. Returns path to saved checkpoint.
    """
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

    model.to(device)

    print(f"Train batches : {len(train_loader)}  Val batches: {len(val_loader)}")

    history: list[dict] = []

    # ── Phase 1: heads only ──────────────────────────────────────────────
    print(f"\n── Phase 1: backbone frozen ({ph1_epochs} epochs, lr={ph1_lr}) ──")
    freeze_backbone(model)
    params = count_params(model)
    print(f"   Trainable params: {params['trainable']:,} / {params['total']:,}")

    optimizer = make_optimizer(model, ph1_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    scaler    = GradScaler()

    run_phase(model, train_loader, optimizer, scheduler, scaler, device,
              ph1_epochs, "phase1", history)

    # ── Phase 2: full fine-tune ──────────────────────────────────────────
    print(f"\n── Phase 2: full fine-tune ({ph2_epochs} epochs, lr={ph2_lr}) ──")
    unfreeze_all(model)
    if mode != "cardd":
        enable_gradient_checkpointing(model)  # memory saving for Jetson; not needed on RTX 4070
    params = count_params(model)
    print(f"   Trainable params: {params['trainable']:,} / {params['total']:,}")

    optimizer = make_optimizer(model, ph2_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ph2_epochs, eta_min=1e-5
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, ph2_epochs + 1):
        t0   = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        row = {"phase": "phase2", "epoch": epoch, **loss, "time_s": round(elapsed, 1)}
        history.append(row)

        comp_str = "  ".join(f"{k}={v:.4f}" for k, v in loss.items() if k != "total")
        print(
            f"[phase2] epoch {epoch:3d}/{ph2_epochs}  "
            f"loss={loss['total']:.4f}  {comp_str}  ({elapsed:.1f}s)"
        )

        torch.cuda.empty_cache()

        if loss["total"] < best_loss:
            best_loss  = loss["total"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save immediately so progress survives an OOM kill
            torch.save({"model_state": best_state, "mode": mode,
                        "best_loss": best_loss, "history": history}, save_path)
            print(f"         ↳ new best loss → checkpoint saved to {save_path}")

    # ── Save ─────────────────────────────────────────────────────────────
    checkpoint = {
        "model_state": best_state,
        "mode":        mode,
        "best_loss":   best_loss,
        "history":     history,
    }
    torch.save(checkpoint, save_path)
    print(f"\nCheckpoint saved → {save_path}")

    # Save training history as JSON for easy inspection
    hist_path = save_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"History saved  → {hist_path}")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parts", "damage", "cardd"], required=True)
    args = parser.parse_args()
    train(args.mode)
