"""
Auto-Owners AI Claims API
Full pipeline: CV detection → cost estimation → LLM explanation + STP decision
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import time

from cv_detector import CVDetector
from cost_estimator import CostEstimator
from audit_logger import log_claim

app = FastAPI(title="Auto-Owners AI Claims API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\nLoading CV models...")
cv_detector = CVDetector()
print("✓ CV models ready")

print("Loading cost estimator...")
cost_estimator = CostEstimator()
print("✓ Cost estimator ready")

# LLM client is optional — gracefully degrades if GEMINI_API_KEY is missing
try:
    from llm_client import LLMClient
    llm_client = LLMClient()
    print("✓ LLM client ready")
    _llm_available = True
except Exception as e:
    print(f"⚠ LLM client unavailable ({e}) — STP decisions will use rule-based fallback")
    llm_client = None
    _llm_available = False

print("Models ready.\n")


def _rule_based_stp(cost_output: dict, detections: list) -> dict:
    """Fallback STP logic when Gemini is unavailable."""
    total_cost = sum(cost_output["total_cost_range"]) / 2
    confidences = [d.get("confidence", 0.5) for d in detections]
    confidence = sum(confidences) / len(confidences) if confidences else 0.5
    severities = [d.get("severity", "minor") for d in detections]
    total_loss = cost_output.get("total_loss", False)

    cost_ok       = total_cost < 1500
    confidence_ok = confidence > 0.70
    severity_ok   = "major" not in severities and "severe" not in severities
    not_total_loss = not total_loss

    stp_eligible = cost_ok and confidence_ok and severity_ok and not_total_loss

    if stp_eligible:
        reasoning = (
            f"Claim eligible for auto-approval: cost ${total_cost:.0f} under $1,500, "
            f"{confidence:.0%} confidence, no major damage, not a total loss."
        )
    else:
        reasons = []
        if total_loss: reasons.append("total loss")
        if not cost_ok: reasons.append(f"cost ${total_cost:.0f} exceeds $1,500")
        if not confidence_ok: reasons.append(f"{confidence:.0%} confidence below 80%")
        if not severity_ok: reasons.append("major/severe damage present")
        reasoning = f"Manual review required: {', '.join(reasons)}."

    requires_review = not stp_eligible or confidence < 0.60 or total_loss

    parts_list = ", ".join({d["part"] for d in detections})
    explanation = (
        f"Our AI detected damage to the following area(s): {parts_list}. "
        f"The estimated repair cost is ${cost_output['total_cost_range'][0]}–"
        f"${cost_output['total_cost_range'][1]}."
    )

    return {
        "damaged_parts":           cost_output["damaged_parts"],
        "total_cost_range":        cost_output["total_cost_range"],
        "explanation":             explanation,
        "confidence_score":        round(confidence, 2),
        "stp_eligible":            stp_eligible,
        "stp_reasoning":           reasoning,
        "requires_adjuster_review": requires_review,
        "total_loss":              total_loss,
        "override_allowed":        True,
        "model_version":           "1.0.0",
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    zipCode: str = Form(default=""),
):
    """
    Upload a vehicle photo. Returns detected parts, damage types, cost estimate,
    and STP decision as JSON.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image provided")

    temp_path = Path(f"temp/temp_{image.filename}")
    temp_path.parent.mkdir(exist_ok=True)

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        t0 = time.perf_counter()

        # ── 1. CV detection ───────────────────────────────────────────────────
        t_cv = time.perf_counter()
        detections = cv_detector.detect(str(temp_path))
        cv_ms = round((time.perf_counter() - t_cv) * 1000, 1)
        print(f"[TIMING] cv_total:      {cv_ms}ms  detections={len(detections)}", flush=True)
        inference_ms = cv_ms

        # ── 2. Cost estimation ────────────────────────────────────────────────
        t_cost = time.perf_counter()
        cost_output = cost_estimator.estimate(detections, zip_code=zipCode)
        cost_ms = round((time.perf_counter() - t_cost) * 1000, 1)
        print(f"[TIMING] cost_estimator:{cost_ms}ms", flush=True)

        # ── 3. LLM explanation + STP decision ─────────────────────────────────
        vehicle_info = {"year": 2021, "make": "Vehicle", "model": ""}
        cv_output_for_llm = {"damaged_parts": detections}

        t_llm = time.perf_counter()
        if _llm_available:
            try:
                llm_output = llm_client.process_claim(cv_output_for_llm, cost_output, vehicle_info)
            except Exception as e:
                print(f"LLM error: {e} — using rule-based fallback")
                llm_output = _rule_based_stp(cost_output, detections)
        else:
            llm_output = _rule_based_stp(cost_output, detections)
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)
        print(f"[TIMING] stp_decision:  {llm_ms}ms  stp={llm_output.get('stp_eligible')}", flush=True)
        print(f"[TIMING] request_total: {round((time.perf_counter() - t0) * 1000, 1)}ms", flush=True)

        # ── 4. Audit trail ────────────────────────────────────────────────────
        try:
            claim_id = log_claim(cv_output_for_llm, cost_output, llm_output)
        except Exception as e:
            print(f"Audit log warning: {e}")
            claim_id = None

        return {
            "detections":              detections,
            "cost":                    cost_output,
            "explanation":             llm_output.get("explanation", ""),
            "confidence_score":        llm_output.get("confidence_score"),
            "stp_eligible":            llm_output.get("stp_eligible"),
            "stp_reasoning":           llm_output.get("stp_reasoning"),
            "requires_adjuster_review": llm_output.get("requires_adjuster_review"),
            "total_loss":              llm_output.get("total_loss", False),
            "override_allowed":        True,
            "model_version":           "1.0.0",
            "fraud_flags":             [],
            "inference_ms":            inference_ms,
            "claim_id":                claim_id,
            # Kept for backwards compatibility with ResultsDisplay
            "summary": {
                "total_damaged_parts": len(detections),
                "parts": list({d["part"] for d in detections}),
                "damage_types": list({d["damage_type"] for d in detections}),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": "Mask R-CNN (parts) + YOLOv8m (damage) + GradientBoosting (cost)",
        "llm": "gemini-1.5-flash" if _llm_available else "unavailable",
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CV Damage Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 700px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #003d7a; }
            input[type=file], input[type=text], button { display: block; width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            button { background: #003d7a; color: white; border: none; cursor: pointer; font-size: 16px; }
            button:hover { background: #005ca8; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CV Damage Detection</h1>
            <p>Mask R-CNN parts + YOLOv8m damage + GradientBoosting cost + Gemini STP</p>
            <form id="form">
                <input type="file" name="image" accept="image/*" required>
                <input type="text" name="zipCode" placeholder="ZIP code (optional)" value="48823">
                <button type="submit">Detect Damage</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById('form').onsubmit = async (e) => {
                e.preventDefault();
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Running detection...</p>';
                const formData = new FormData(e.target);
                try {
                    const res = await fetch('/detect', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (!res.ok) { resultDiv.innerHTML = `<p><b>Error:</b> ${data.detail}</p>`; return; }
                    resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                } catch (err) {
                    resultDiv.innerHTML = `<p><b>Error:</b> ${err.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
