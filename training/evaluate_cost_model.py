"""
Cost model evaluation — held-out test set.

Generates 500 samples using the same synthetic data process as training
(but with a different random seed so they were never seen by the model),
then reports RMSE, MAE, R², and mean percentage error.

Run from the backend directory:
    python evaluate_cost_model.py
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from cost_estimator import (
    _generate_training_data,
    BASE_COSTS,
    COST_MODEL_PATH,
    ENCODERS_PATH,
    _train_and_save,
)

# ── Generate held-out test set (different seed from training) ─────────────────

def _generate_test_data(n: int = 500) -> list:
    """Same logic as _generate_training_data but seed=99 (never seen by model)."""
    from cost_estimator import DAMAGE_TYPES, SEVERITIES, SEVERITY_MULTIPLIER, _get_action
    rng   = np.random.default_rng(99)   # different seed — model trained on seed=42
    parts = list(BASE_COSTS.keys())
    rows  = []
    for _ in range(n):
        part        = rng.choice(parts)
        damage_type = rng.choice(DAMAGE_TYPES)
        severity    = rng.choice(SEVERITIES)
        action      = _get_action(part, damage_type, severity)
        repair_cost, replace_cost = BASE_COSTS[part]
        base = replace_cost if action == "replace" else (repair_cost or replace_cost)
        cost = base * SEVERITY_MULTIPLIER[severity] * rng.uniform(0.85, 1.15)
        rows.append((part, damage_type, severity, float(cost)))
    return rows


def evaluate():
    if not COST_MODEL_PATH.exists():
        print("No model found — training first...")
        _train_and_save()

    model    = joblib.load(COST_MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    le_part  = encoders["part"]
    le_dmg   = encoders["damage_type"]
    le_sev   = encoders["severity"]

    print("Generating 500 held-out test samples (seed=99, model trained on seed=42)...")
    test_data = _generate_test_data(500)
    parts_col, dmg_col, sev_col, cost_col = zip(*test_data)

    X_test = np.column_stack([
        le_part.transform(parts_col),
        le_dmg.transform(dmg_col),
        le_sev.transform(sev_col),
    ])
    y_true = np.array(cost_col)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n── Cost Model Evaluation (n=500 held-out samples) ──")
    print(f"  RMSE  : ${rmse:,.2f}   (avg prediction error in dollars)")
    print(f"  MAE   : ${mae:,.2f}   (avg absolute error)")
    print(f"  R²    : {r2:.4f}     (1.0 = perfect, >0.90 = strong)")
    print(f"  MAPE  : {mape:.2f}%   (mean absolute percentage error)")
    print(f"\n── Per-severity breakdown ──")
    for sev in ["minor", "moderate", "severe"]:
        mask  = np.array(sev_col) == sev
        if mask.sum() == 0:
            continue
        r2_s  = r2_score(y_true[mask], y_pred[mask])
        mae_s = mean_absolute_error(y_true[mask], y_pred[mask])
        print(f"  {sev:<10} n={mask.sum():>3}   MAE=${mae_s:>7,.2f}   R²={r2_s:.4f}")

    print(f"\n── Sample predictions vs actuals ──")
    print(f"  {'Part':<20} {'Damage':<14} {'Sev':<10} {'Actual':>10} {'Predicted':>10} {'Error%':>8}")
    idxs = np.random.default_rng(0).choice(len(test_data), 10, replace=False)
    for i in idxs:
        part, dmg, sev, actual = test_data[i]
        pred = y_pred[i]
        err  = abs(actual - pred) / actual * 100
        print(f"  {part:<20} {dmg:<14} {sev:<10} ${actual:>8,.0f} ${pred:>8,.0f} {err:>7.1f}%")


if __name__ == "__main__":
    evaluate()
