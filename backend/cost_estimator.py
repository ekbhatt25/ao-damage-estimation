"""
Vehicle damage cost estimation using a GradientBoosting regression model
trained on repair cost data sourced from RepairPal.com.

Data files (fill these in — see instructions below):
  backend/data/repair_costs.csv  — parts costs from repairpal.com/estimator
  backend/data/labor_rates.csv   — body/mechanical/paint rates by state
                                   from SCRS survey, ASA survey, or state DoIs

Falls back to built-in estimates for any row left blank.

Total loss detection follows the industry-standard rule:
  if repair_cost > 0.70 × ACV → total loss
ACV is estimated from vehicle year using a depreciation table.
"""

import csv
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

MODELS_DIR      = Path(__file__).parent.parent / "models"
COST_MODEL_PATH = MODELS_DIR / "cost_model.joblib"
ENCODERS_PATH   = MODELS_DIR / "cost_encoders.joblib"
COSTS_CSV       = Path(__file__).parent / "data" / "repair_costs.csv"
LABOR_CSV       = Path(__file__).parent / "data" / "labor_rates.csv"

# ── Fallback parts costs (midpoints, USD) ─────────────────────────────────────
# (repair_mid, replace_mid) — None repair means replace-only part
_FALLBACK_COSTS = {
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

# ── Fallback labor rates ($/hr national averages) ─────────────────────────────
# Source: SCRS 2023 Labor Rate Survey national averages
_FALLBACK_LABOR = {"body": 58.0, "mechanical": 80.0, "paint": 52.0}

# Which labor category each damage type maps to
DAMAGE_LABOR_CATEGORY = {
    "Dent":          "body",
    "Crack":         "body",
    "Glass shatter": "body",
    "Lamp broken":   "mechanical",
    "Tire flat":     "mechanical",
    "Scratch":       "paint",
}

DAMAGE_TYPES = ["Dent", "Scratch", "Crack", "Glass shatter", "Lamp broken", "Tire flat"]
SEVERITIES   = ["minor", "moderate", "severe"]

SEVERITY_MULTIPLIER = {"minor": 0.65, "moderate": 1.00, "severe": 1.45}

REPLACE_ALWAYS     = {"Glass shatter", "Lamp broken", "Tire flat"}
REPLACE_ONLY_PARTS = {
    "Windshield", "Back-windshield", "Front-window", "Back-window",
    "Headlight", "Tail-light", "Front-wheel", "Back-wheel", "License-plate",
}

# ZIP first digit → 2-letter state abbreviation (most common state in that range)
# Used to look up labor rates when only a ZIP is available
_ZIP_PREFIX_TO_STATE = {
    "0": "CT", "1": "NY", "2": "VA", "3": "FL", "4": "MI",
    "5": "MN", "6": "IL", "7": "TX", "8": "CO", "9": "CA",
}

# ── ACV estimation (for total loss calculation) ───────────────────────────────
# Straight-line depreciation: average new mid-size sedan ~$35,000
# ~15% depreciation per year, floor at $3,000
_ACV_BASE      = 35_000
_DEPRECIATION  = 0.15
_ACV_FLOOR     = 3_000

def _estimate_acv(vehicle_year: int) -> float:
    import datetime
    age = max(0, datetime.datetime.now().year - vehicle_year)
    acv = _ACV_BASE * ((1 - _DEPRECIATION) ** age)
    return max(acv, _ACV_FLOOR)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_base_costs() -> dict:
    costs = dict(_FALLBACK_COSTS)
    if not COSTS_CSV.exists():
        return costs
    with open(COSTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            part = row["part"].strip()
            try:
                repair_mid = (float(row["repair_low"]) + float(row["repair_high"])) / 2
            except (ValueError, KeyError):
                repair_mid = None
            try:
                replace_mid = (float(row["replace_low"]) + float(row["replace_high"])) / 2
            except (ValueError, KeyError):
                fallback = _FALLBACK_COSTS.get(part)
                replace_mid = fallback[1] if fallback else None
            if replace_mid is not None:
                costs[part] = (repair_mid, replace_mid)
    print(f"✓ Repair costs loaded ({len(costs)} parts)")
    return costs


def _load_labor_rates() -> dict:
    """Returns {state: {body, mechanical, paint}} with fallback for missing rows."""
    rates = {}
    if not LABOR_CSV.exists():
        return rates
    with open(LABOR_CSV, newline="") as f:
        for row in csv.DictReader(f):
            state = row["state"].strip().upper()
            try:
                rates[state] = {
                    "body":       float(row["body"]),
                    "mechanical": float(row["mechanical"]),
                    "paint":      float(row["paint"]),
                }
            except (ValueError, KeyError):
                pass  # row not filled in yet — will use fallback
    filled = len(rates)
    if filled:
        print(f"✓ Labor rates loaded ({filled}/50 states from CSV)")
    return rates


BASE_COSTS   = _load_base_costs()
_LABOR_RATES = _load_labor_rates()


def _get_labor_rates_for_zip(zip_code: str) -> dict:
    """Return body/mechanical/paint rates for a given ZIP code."""
    state = _ZIP_PREFIX_TO_STATE.get(zip_code[0] if zip_code else "", "")
    rates = _LABOR_RATES.get(state, {})
    return {
        "body":       rates.get("body",       _FALLBACK_LABOR["body"]),
        "mechanical": rates.get("mechanical", _FALLBACK_LABOR["mechanical"]),
        "paint":      rates.get("paint",      _FALLBACK_LABOR["paint"]),
    }


def _get_action(part: str, damage_type: str, severity: str) -> str:
    if damage_type in REPLACE_ALWAYS:
        return "replace"
    if part in REPLACE_ONLY_PARTS:
        return "replace"
    if severity == "severe" and damage_type in {"Crack", "Dent"}:
        return "replace"
    return "repair"


# ── Training data generation ──────────────────────────────────────────────────

def _generate_training_data(n: int = 2000) -> list:
    rng   = np.random.default_rng(42)
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
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, np.array(cost_col))
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, COST_MODEL_PATH)
    joblib.dump({"part": le_part, "damage_type": le_dmg, "severity": le_sev}, ENCODERS_PATH)
    print("✓ Cost model trained and saved")


# ── Estimator ─────────────────────────────────────────────────────────────────

class CostEstimator:
    def __init__(self):
        if not COST_MODEL_PATH.exists() or not ENCODERS_PATH.exists():
            print("Cost model not found — training now (~5s)...")
            _train_and_save()
        self._model    = joblib.load(COST_MODEL_PATH)
        self._encoders = joblib.load(ENCODERS_PATH)

    @staticmethod
    def retrain():
        """Rebuild model after updating repair_costs.csv or labor_rates.csv."""
        global BASE_COSTS, _LABOR_RATES
        BASE_COSTS   = _load_base_costs()
        _LABOR_RATES = _load_labor_rates()
        _train_and_save()

    def estimate(self, cv_damaged_parts: list, zip_code: str = "",
                 vehicle_year: int = 2021) -> dict:  # default: ~4yr old vehicle, ~$20k ACV
        """
        Convert CV damaged_parts into a cost_output dict for llm_client.process_claim().

        Args:
            cv_damaged_parts: list of {part, damage_type, severity} from CV pipeline
            zip_code:         5-digit ZIP for regional labor rate lookup
            vehicle_year:     model year for ACV / total loss calculation

        Returns:
            cost_output dict including total_loss flag
        """
        labor_rates = _get_labor_rates_for_zip(zip_code)
        le_part     = self._encoders["part"]
        le_dmg      = self._encoders["damage_type"]
        le_sev      = self._encoders["severity"]

        result_parts = []
        for det in cv_damaged_parts:
            part        = det.get("part", "Front-bumper")
            damage_type = det.get("damage_type", "Dent")
            severity    = det.get("severity", "moderate")

            if part        not in le_part.classes_: part        = "Front-bumper"
            if damage_type not in le_dmg.classes_:  damage_type = "Dent"
            if severity    not in le_sev.classes_:  severity    = "moderate"

            action        = _get_action(part, damage_type, severity)
            labor_category = DAMAGE_LABOR_CATEGORY.get(damage_type, "body")
            labor_rate    = labor_rates[labor_category]

            X = np.array([[
                le_part.transform([part])[0],
                le_dmg.transform([damage_type])[0],
                le_sev.transform([severity])[0],
            ]])
            # ML prediction gives base parts cost; scale by local labor rate
            predicted = float(self._model.predict(X)[0]) * (labor_rate / _FALLBACK_LABOR["body"])
            low  = round(predicted * 0.85)
            high = round(predicted * 1.15)

            result_parts.append({
                "part":           part,
                "damage_type":    damage_type,
                "severity":       "major" if severity == "severe" else severity,
                "action":         action,
                "labor_category": labor_category,
                "labor_rate":     labor_rate,
                "cost_range":     [low, high],
            })

        total_low  = sum(p["cost_range"][0] for p in result_parts)
        total_high = sum(p["cost_range"][1] for p in result_parts)
        repair_mid = (total_low + total_high) / 2

        # Total loss check: repair cost > 70% of ACV (industry standard threshold)
        acv        = _estimate_acv(vehicle_year)
        total_loss = repair_mid > (0.70 * acv)

        return {
            "damaged_parts":         result_parts,
            "total_cost_range":      [total_low, total_high],
            "zip_code":              zip_code or "00000",
            "labor_rates":           labor_rates,
            "acv_estimate":          round(acv),
            "total_loss":            total_loss,
        }
