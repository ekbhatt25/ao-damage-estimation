"""
Vehicle damage cost estimation using a GradientBoosting regression model
trained on synthetic data derived from public repair cost averages
(RepairPal, AAA Automotive repair guides).

The model takes (part, damage_type, severity) and predicts repair cost.
Training data is generated at first startup — no external dataset needed.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path(__file__).parent.parent / "models"
COST_MODEL_PATH   = MODELS_DIR / "cost_model.joblib"
ENCODERS_PATH     = MODELS_DIR / "cost_encoders.joblib"

# ── Base repair costs (national average USD) ──────────────────────────────────
# Source: RepairPal.com / AAA Automotive repair guides
# Tuple: (repair_cost, replace_cost)  —  None means part can only be replaced
BASE_COSTS = {
    "Front-bumper":    (350,  900),
    "Back-bumper":     (300,  800),
    "Hood":            (600, 1200),
    "Front-door":      (500, 1100),
    "Back-door":       (450, 1000),
    "Fender":          (400,  850),
    "Windshield":      (None, 350),
    "Back-windshield": (None, 300),
    "Front-window":    (None, 200),
    "Back-window":     (None, 200),
    "Headlight":       (None, 275),
    "Tail-light":      (None, 175),
    "Mirror":          (None, 200),
    "Grille":          (150,  250),
    "Roof":            (700, 2000),
    "Trunk":           (400,  900),
    "Quarter-panel":   (600, 1100),
    "Rocker-panel":    (300,  600),
    "Front-wheel":     (None, 250),
    "Back-wheel":      (None, 250),
    "License-plate":   (None,  50),
}

DAMAGE_TYPES = ["Dent", "Scratch", "Crack", "Glass shatter", "Lamp broken", "Tire flat"]
SEVERITIES   = ["minor", "moderate", "severe"]

SEVERITY_MULTIPLIER = {"minor": 0.65, "moderate": 1.00, "severe": 1.45}

# Damage types that always force a replace regardless of severity
REPLACE_ALWAYS = {"Glass shatter", "Lamp broken", "Tire flat"}

# Parts that have no repair option
REPLACE_ONLY_PARTS = {
    "Windshield", "Back-windshield", "Front-window", "Back-window",
    "Headlight", "Tail-light", "Front-wheel", "Back-wheel", "License-plate",
}

# Regional labor rate multipliers keyed on first digit of ZIP code
# Reflects Bureau of Labor Statistics regional wage data
LABOR_MULTIPLIERS = {
    "0": 1.15,  # CT, MA, ME, NH, NJ, NY, RI, VT
    "1": 1.10,  # DE, NY, PA
    "2": 1.00,  # DC, MD, NC, SC, VA, WV
    "3": 0.95,  # AL, FL, GA, MS, TN
    "4": 0.90,  # IN, KY, MI, OH
    "5": 0.88,  # IA, MN, MT, ND, SD, WI
    "6": 1.05,  # IL, KS, MO, NE
    "7": 0.90,  # AR, LA, OK, TX
    "8": 0.95,  # AZ, CO, ID, NM, NV, UT, WY
    "9": 1.10,  # AK, CA, HI, OR, WA
}


def _get_action(part: str, damage_type: str, severity: str) -> str:
    if damage_type in REPLACE_ALWAYS:
        return "replace"
    if part in REPLACE_ONLY_PARTS:
        return "replace"
    if severity == "severe" and damage_type in {"Crack", "Dent"}:
        return "replace"
    return "repair"


def _get_labor_multiplier(zip_code: str) -> float:
    if zip_code and len(zip_code) >= 1 and zip_code[0].isdigit():
        return LABOR_MULTIPLIERS.get(zip_code[0], 1.0)
    return 1.0


# ── Synthetic training data ───────────────────────────────────────────────────

def _generate_training_data(n: int = 2000) -> list:
    rng = np.random.default_rng(42)
    parts = list(BASE_COSTS.keys())
    rows = []

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


# ── Training ──────────────────────────────────────────────────────────────────

def _train_and_save():
    data = _generate_training_data()
    parts_col, dmg_col, sev_col, cost_col = zip(*data)

    le_part = LabelEncoder().fit(list(BASE_COSTS.keys()))
    le_dmg  = LabelEncoder().fit(DAMAGE_TYPES)
    le_sev  = LabelEncoder().fit(SEVERITIES)

    X = np.column_stack([
        le_part.transform(parts_col),
        le_dmg.transform(dmg_col),
        le_sev.transform(sev_col),
    ])
    y = np.array(cost_col)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, COST_MODEL_PATH)
    joblib.dump({"part": le_part, "damage_type": le_dmg, "severity": le_sev}, ENCODERS_PATH)
    print("✓ Cost model trained and saved")


# ── Estimator ─────────────────────────────────────────────────────────────────

class CostEstimator:
    def __init__(self):
        if not COST_MODEL_PATH.exists() or not ENCODERS_PATH.exists():
            print("Cost model not found — training now (takes ~5s)...")
            _train_and_save()
        self._model    = joblib.load(COST_MODEL_PATH)
        self._encoders = joblib.load(ENCODERS_PATH)

    def estimate(self, cv_damaged_parts: list, zip_code: str = "") -> dict:
        """
        Convert CV damaged_parts detections into a cost_output dict
        compatible with llm_client.LLMClient.process_claim().

        Args:
            cv_damaged_parts: list of dicts from the CV pipeline, each with
                              keys: part, damage_type, severity
            zip_code:         5-digit ZIP for regional labor rate adjustment

        Returns:
            cost_output dict with keys:
                damaged_parts, total_cost_range, zip_code, labor_rate_multiplier
        """
        labor    = _get_labor_multiplier(zip_code)
        le_part  = self._encoders["part"]
        le_dmg   = self._encoders["damage_type"]
        le_sev   = self._encoders["severity"]

        result_parts = []
        for det in cv_damaged_parts:
            part        = det.get("part", "Front-bumper")
            damage_type = det.get("damage_type", "Dent")
            severity    = det.get("severity", "moderate")

            # Clamp unknowns to nearest valid class
            if part not in le_part.classes_:
                part = "Front-bumper"
            if damage_type not in le_dmg.classes_:
                damage_type = "Dent"
            if severity not in le_sev.classes_:
                severity = "moderate"

            action = _get_action(part, damage_type, severity)

            X = np.array([[
                le_part.transform([part])[0],
                le_dmg.transform([damage_type])[0],
                le_sev.transform([severity])[0],
            ]])
            predicted = float(self._model.predict(X)[0]) * labor
            low  = round(predicted * 0.85)
            high = round(predicted * 1.15)

            result_parts.append({
                "part":        part,
                "damage_type": damage_type,
                # llm_client uses "major" where CV uses "severe"
                "severity":    "major" if severity == "severe" else severity,
                "action":      action,
                "cost_range":  [low, high],
            })

        total_low  = sum(p["cost_range"][0] for p in result_parts)
        total_high = sum(p["cost_range"][1] for p in result_parts)

        return {
            "damaged_parts":         result_parts,
            "total_cost_range":      [total_low, total_high],
            "zip_code":              zip_code or "00000",
            "labor_rate_multiplier": labor,
        }
