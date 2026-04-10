"""
Audit trail for every claim processed through the pipeline.
Logs model version, detections, confidence, cost, and STP decision
to satisfy Sprint 4 auditability requirements.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = Path(__file__).parent.parent / "audit_log.jsonl"


def log_claim(cv_output: dict, cost_output: dict, llm_output: dict) -> str:
    """
    Write one audit record per claim. Returns the claim_id.
    Each line in audit_log.jsonl is a self-contained JSON record.
    """
    claim_id = str(uuid.uuid4())

    record = {
        "claim_id":              claim_id,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "model_version":         llm_output.get("model_version", "1.0.0"),
        "damaged_parts":         llm_output.get("damaged_parts", []),
        "total_cost_range":      llm_output.get("total_cost_range", []),
        "confidence_score":      llm_output.get("confidence_score"),
        "stp_eligible":          llm_output.get("stp_eligible"),
        "stp_reasoning":         llm_output.get("stp_reasoning"),
        "requires_adjuster_review": llm_output.get("requires_adjuster_review"),
        "total_loss":            llm_output.get("total_loss", False),
        "override_allowed":      llm_output.get("override_allowed", True),
        "explanation":           llm_output.get("explanation", ""),
        "acv_estimate":          cost_output.get("acv_estimate"),
        "zip_code":              cost_output.get("zip_code"),
        "labor_rates":           cost_output.get("labor_rates"),
        "cv_inference_parts":    len(cv_output.get("damaged_parts", [])),
    }

    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return claim_id
