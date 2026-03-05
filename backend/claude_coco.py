"""
claude_coco.py — Claude Sonnet 4.6 Evaluation Harness

Runs the standardized insurance assessment prompt (from test_llm.py) against
all COCO train + val cases and records:
  - Raw JSON output from the model
  - Whether output is valid JSON
  - Latency per call
  - Consistency: 5 sample cases run 3x each
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from test_llm import generate_prompt

load_dotenv()

MODEL = "claude-sonnet-4-6"
CONSISTENCY_SAMPLE = 5
CONSISTENCY_RUNS = 3
RANDOM_SEED = 42

REQUIRED_FIELDS = [
    "damaged_parts",
    "total_cost_range",
    "explanation",
    "confidence_score",
    "stp_eligible",
    "stp_reasoning",
]


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
            cases.append(
                {
                    "image_id": img["id"],
                    "filename": img["file_name"],
                    "damaged_parts": sorted(parts),
                }
            )
    return cases


def _extract_text(response: anthropic.types.Message) -> str:
    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "\n".join(text_blocks).strip()


def _parse_json_output(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return json.loads(cleaned)


def call_claude(prompt: str) -> tuple:
    """Returns (result_dict | None, latency_seconds, json_valid_bool)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    t0 = time.time()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - t0
        raw_text = _extract_text(response)
        result = _parse_json_output(raw_text)
        valid = isinstance(result, dict) and all(f in result for f in REQUIRED_FIELDS)
        return result, latency, valid
    except Exception as e:
        print(f"    [API error] {type(e).__name__}: {e}")
        return None, time.time() - t0, False


def run_benchmark(cases: list[dict], split_name: str) -> list[dict]:
    results = []
    print(f"\n-- {split_name.upper()} ({len(cases)} images) --")
    for case in cases:
        prompt, _ = generate_prompt(case["damaged_parts"])
        result, latency, json_ok = call_claude(prompt)
        status = "OK" if json_ok else "ERR"
        print(f"  {status} {case['filename']:<12} | {latency:.2f}s | valid={json_ok}")
        results.append(
            {
                "split": split_name,
                "filename": case["filename"],
                "input_parts": case["damaged_parts"],
                "json_valid": json_ok,
                "latency": round(latency, 2),
                "output": result,
            }
        )
    return results


def run_consistency(all_cases: list[dict]) -> list[dict]:
    random.seed(RANDOM_SEED)
    sample = random.sample(all_cases, min(CONSISTENCY_SAMPLE, len(all_cases)))
    print(f"\n-- CONSISTENCY TEST ({len(sample)} images x {CONSISTENCY_RUNS} runs) --")
    rows = []
    for case in sample:
        prompt, _ = generate_prompt(case["damaged_parts"])
        runs = []
        for i in range(CONSISTENCY_RUNS):
            result, lat, _ = call_claude(prompt)
            runs.append({"result": result, "latency": round(lat, 2)})
            print(f"  {case['filename']} run {i + 1}: {lat:.2f}s")
        rows.append(
            {
                "filename": case["filename"],
                "input_parts": case["damaged_parts"],
                "runs": runs,
            }
        )
    return rows


def print_report(all_results: list[dict], consistency_rows: list[dict]):
    w = 80
    valid_count = sum(1 for r in all_results if r["json_valid"])
    latencies = [r["latency"] for r in all_results]

    print("\n" + "=" * w)
    print("CLAUDE SONNET 4.6 -- EVALUATION REPORT")
    print("=" * w)

    print(
        f"""
API ACCESS & SECURITY
----------------------
Model:            {MODEL}
Provider:         Anthropic
SDK:              anthropic (Python)

Authentication:   API key via ANTHROPIC_API_KEY environment variable
Key storage:      .env file (python-dotenv); .env is gitignored and never committed
Key management:   Rotate/revoke keys in Anthropic Console
                  For production: use a managed secret store

Rate limits:      Vary by Anthropic plan and model tier (check console)
Data retention:   Subject to Anthropic API policy and account settings
HIPAA:            Verify eligibility and legal agreements with Anthropic
SOC 2 / ISO:      See Anthropic trust/compliance documentation
"""
    )

    print("SUMMARY")
    print("-------")
    print(f"Model:             {MODEL}")
    print(
        f"Total cases:       {len(all_results)}  "
        f"(train={sum(1 for r in all_results if r['split'] == 'train')}, "
        f"val={sum(1 for r in all_results if r['split'] == 'val')})"
    )
    print(f"JSON valid:        {valid_count}/{len(all_results)}")
    print(f"Avg latency:       {sum(latencies)/len(latencies):.2f}s")
    print(f"Min/Max latency:   {min(latencies):.2f}s / {max(latencies):.2f}s")

    print("\n\nPER-IMAGE OUTPUTS")
    print("=" * w)
    for r in all_results:
        print(
            f"\n[{r['split'].upper()}] {r['filename']}  |  "
            f"latency={r['latency']}s  |  json_valid={r['json_valid']}"
        )
        print(f"Input parts: {', '.join(r['input_parts'])}")
        print("Output:")
        print(json.dumps(r["output"], indent=2) if r["output"] else "  ERROR -- no output")
        print("-" * w)

    print(f"\n\nCONSISTENCY TEST  ({CONSISTENCY_SAMPLE} images x {CONSISTENCY_RUNS} runs)")
    print("=" * w)
    for r in consistency_rows:
        print(f"\n{r['filename']}  |  input_parts: {', '.join(r['input_parts'])}")
        for i, run in enumerate(r["runs"], 1):
            print(f"\n  Run {i}  ({run['latency']}s):")
            print(json.dumps(run["result"], indent=4) if run["result"] else "    ERROR -- no output")
        print("-" * w)

    print("\n" + "=" * w)
    print("END OF REPORT")
    print("=" * w)


if __name__ == "__main__":
    import sys as _sys

    out_path = Path(__file__).parent / "claude46_results.txt"
    with open(out_path, "w") as f:
        class Tee:
            def write(self, msg):
                _sys.__stdout__.write(msg)
                f.write(msg)
                f.flush()

            def flush(self):
                _sys.__stdout__.flush()
                f.flush()

        _sys.stdout = Tee()
        try:
            root = Path(__file__).parent.parent

            print(f"Writing live output to {out_path}")

            train_cases = load_split(root / "archive" / "train" / "COCO_mul_train_annos.json")
            val_cases = load_split(root / "archive" / "val" / "COCO_mul_val_annos.json")
            all_cases = train_cases + val_cases

            print(f"Loaded {len(train_cases)} train + {len(val_cases)} val = {len(all_cases)} total cases")

            train_results = run_benchmark(train_cases, "train")
            val_results = run_benchmark(val_cases, "val")
            all_results = train_results + val_results

            consistency_rows = run_consistency(all_cases)
            print_report(all_results, consistency_rows)
        finally:
            _sys.stdout = _sys.__stdout__

    print(f"\nReport saved to {out_path}")
