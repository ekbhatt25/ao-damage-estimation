"""
gemini_coco.py — Gemini 3 Flash Preview Evaluation Harness

Runs the standardized insurance assessment prompt (from test_llm.py) against
all COCO train + val cases and records:
  - Raw JSON output from the model
  - Whether output is valid JSON
  - Latency per call
  - Consistency: 5 sample cases run 3x each
"""

import json
import time
import random
import sys
from pathlib import Path
from google import genai
import os
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from test_llm import generate_prompt

load_dotenv()

MODEL             = "gemini-3-flash-preview"
CONSISTENCY_SAMPLE = 5
CONSISTENCY_RUNS  = 3
RANDOM_SEED       = 42

REQUIRED_FIELDS = [
    "damaged_parts", "total_cost_range", "explanation",
    "confidence_score", "stp_eligible", "stp_reasoning",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "damaged_parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part":        {"type": "string"},
                    "damage_type": {"type": "string"},
                    "severity":    {"type": "string"},
                    "action":      {"type": "string"},
                    "cost_range":  {"type": "array", "items": {"type": "number"}},
                },
                "required": ["part", "damage_type", "severity", "action", "cost_range"],
            },
        },
        "total_cost_range":  {"type": "array", "items": {"type": "number"}},
        "explanation":       {"type": "string"},
        "confidence_score":  {"type": "number"},
        "stp_eligible":      {"type": "boolean"},
        "stp_reasoning":     {"type": "string"},
    },
    "required": REQUIRED_FIELDS,
}


def load_split(json_path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    cats = {c["id"]: c["name"].lower().replace(" ", "_") for c in data["categories"]}
    grouped = {}
    for a in data["annotations"]:
        grouped.setdefault(a["image_id"], set()).add(cats.get(a["category_id"]))
    cases = []
    for img in data["images"]:
        parts = grouped.get(img["id"])
        if parts:
            cases.append({
                "image_id":      img["id"],
                "filename":      img["file_name"],
                "damaged_parts": sorted(parts),
            })
    return cases


def call_gemini3(prompt: str) -> tuple:
    """Returns (result_dict | None, latency_seconds, json_valid_bool)."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema":    RESPONSE_SCHEMA,
                "max_output_tokens":  8192,
                "thinking_config":    {"thinking_level": "minimal"},
            },
        )
        latency = time.time() - t0
        result  = json.loads(response.text.strip())
        valid   = isinstance(result, dict) and all(f in result for f in REQUIRED_FIELDS)
        return result, latency, valid
    except Exception as e:
        print(f"    [API error] {type(e).__name__}: {e}")
        return None, time.time() - t0, False


def run_benchmark(cases: list[dict], split_name: str) -> list[dict]:
    results = []
    print(f"\n── {split_name.upper()} ({len(cases)} images) ──")
    for case in cases:
        prompt, _ = generate_prompt(case["damaged_parts"])
        result, latency, json_ok = call_gemini3(prompt)
        status = "✓" if json_ok else "✗"
        print(f"  {status} {case['filename']:<12} | {latency:.2f}s | valid={json_ok}")
        results.append({
            "split":        split_name,
            "filename":     case["filename"],
            "input_parts":  case["damaged_parts"],
            "json_valid":   json_ok,
            "latency":      round(latency, 2),
            "output":       result,
        })
    return results


def run_consistency(all_cases: list[dict]) -> list[dict]:
    random.seed(RANDOM_SEED)
    sample = random.sample(all_cases, min(CONSISTENCY_SAMPLE, len(all_cases)))
    print(f"\n── CONSISTENCY TEST ({len(sample)} images × {CONSISTENCY_RUNS} runs) ──")
    rows = []
    for case in sample:
        prompt, _ = generate_prompt(case["damaged_parts"])
        runs = []
        for i in range(CONSISTENCY_RUNS):
            result, lat, _ = call_gemini3(prompt)
            runs.append({"result": result, "latency": round(lat, 2)})
            print(f"  {case['filename']} run {i + 1}: {lat:.2f}s")
        rows.append({
            "filename":    case["filename"],
            "input_parts": case["damaged_parts"],
            "runs":        runs,
        })
    return rows


def print_report(all_results: list[dict], consistency_rows: list[dict]):
    W = 80
    valid_count = sum(1 for r in all_results if r["json_valid"])
    latencies   = [r["latency"] for r in all_results]

    print("\n" + "=" * W)
    print("GEMINI 3 FLASH PREVIEW — EVALUATION REPORT")
    print("=" * W)

    print(f"""
API ACCESS & SECURITY
----------------------
Model:            {MODEL}
Provider:         Google DeepMind / Google AI Studio
SDK:              google-genai (Python)

Authentication:   API key via GEMINI_API_KEY environment variable
Key storage:      .env file (python-dotenv); .env is gitignored and never committed
Key management:   Rotate/revoke at https://aistudio.google.com/app/apikey
                  For production: use Google Cloud Secret Manager

Rate limits (preview tier — check live at https://aistudio.google.com/rate-limit):
  Free tier:      ~10 RPM / 250 RPD
  Paid tier 1:    ~50 RPM / 1,000 RPD
  Note: Preview model limits may be lower than documented; verify in your dashboard

Data retention:   API inputs/outputs NOT used to train Google models (API key path)
HIPAA:            Supported via Google Cloud BAA — must be explicitly configured
                  Free/consumer tier is NOT HIPAA-covered
SOC 2:            SOC 1/2/3 certified (Google Cloud infrastructure)
ISO:              ISO 27001, ISO 42001 (AI Management Systems)
FedRAMP:          FedRAMP High authorized (via Vertex AI)
Enterprise note:  For HIPAA/regulated workloads, use Vertex AI not the free API
""")

    print("SUMMARY")
    print("-------")
    print(f"Model:             {MODEL}")
    print(f"Total cases:       {len(all_results)}  (train={sum(1 for r in all_results if r['split']=='train')}, val={sum(1 for r in all_results if r['split']=='val')})")
    print(f"JSON valid:        {valid_count}/{len(all_results)}")
    print(f"Avg latency:       {sum(latencies)/len(latencies):.2f}s")
    print(f"Min/Max latency:   {min(latencies):.2f}s / {max(latencies):.2f}s")

    print(f"\n\nPER-IMAGE OUTPUTS")
    print("=" * W)
    for r in all_results:
        print(f"\n[{r['split'].upper()}] {r['filename']}  |  latency={r['latency']}s  |  json_valid={r['json_valid']}")
        print(f"Input parts: {', '.join(r['input_parts'])}")
        print("Output:")
        print(json.dumps(r["output"], indent=2) if r["output"] else "  ERROR — no output")
        print("-" * W)

    print(f"\n\nCONSISTENCY TEST  ({CONSISTENCY_SAMPLE} images × {CONSISTENCY_RUNS} runs)")
    print("=" * W)
    for r in consistency_rows:
        print(f"\n{r['filename']}  |  input_parts: {', '.join(r['input_parts'])}")
        for i, run in enumerate(r["runs"], 1):
            print(f"\n  Run {i}  ({run['latency']}s):")
            print(json.dumps(run["result"], indent=4) if run["result"] else "    ERROR — no output")
        print("-" * W)

    print("\n" + "=" * W)
    print("END OF REPORT")
    print("=" * W)


if __name__ == "__main__":
    import sys as _sys

    root = Path(__file__).parent.parent

    train_cases = load_split(root / "archive" / "train" / "COCO_mul_train_annos.json")
    val_cases   = load_split(root / "archive" / "val"   / "COCO_mul_val_annos.json")
    all_cases   = train_cases + val_cases

    print(f"Loaded {len(train_cases)} train + {len(val_cases)} val = {len(all_cases)} total cases")

    train_results = run_benchmark(train_cases, "train")
    val_results   = run_benchmark(val_cases,   "val")
    all_results   = train_results + val_results

    consistency_rows = run_consistency(all_cases)

    # Write report to both stdout and a txt file
    out_path = Path(__file__).parent / "gemini3_results.txt"
    with open(out_path, "w") as f:
        class Tee:
            def write(self, msg):
                _sys.__stdout__.write(msg)
                f.write(msg)
            def flush(self):
                _sys.__stdout__.flush()
                f.flush()
        _sys.stdout = Tee()
        print_report(all_results, consistency_rows)
        _sys.stdout = _sys.__stdout__

    print(f"\nReport saved to {out_path}")
