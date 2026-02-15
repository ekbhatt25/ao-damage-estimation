import json, base64
from pathlib import Path
import anthropic
import os
from dotenv import load_dotenv

# load env variables
load_dotenv()


def analyze_damage(image_path):
    """
    Inputs image and returns a damaged parts list.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Match OpenAI prompt exactly for fair comparison
    system_prompt = "You are a professional car damage classifier. Return ONLY JSON: {'damaged_parts': [{'part': 'name'}]}"
    user_prompt = "Identify damage to: headlamp, front_bumper, hood, door, rear_bumper. Return ONLY JSON."

    # Define JSON schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "damaged_parts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"part": {"type": "string"}},
                    "required": ["part"],
                },
            }
        },
        "required": ["damaged_parts"],
    }

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens = 300,
            system = system_prompt,  # Change model here to test different ones
            messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
            ],
        }
    ],
    tools = [{"name": "damaged_parts", "description":"...", "input_schema": schema}],
    tool_choice = {"type": "tool", "name": "damaged_parts"}
        )

        tool_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None
        )
        if tool_block is None:
            raise ValueError("No tool_use block in Claude response")

        result = tool_block.input
        parts_list = result.get("damaged_parts", [])
        
        cleaned = []
        for item in parts_list:
            if isinstance(item, str): cleaned.append(item)
            elif isinstance(item, dict) and "part" in item: cleaned.append(item["part"])
        return set(cleaned)
    except Exception as e:
        print(f"[Claude error] {type(e).__name__}: {e}")
        return set()


def run_benchmark(image_folder, labels_file, split_name):
    """
    Runs against JSON labels to calculate accuracy.
    """
    image_dir = Path(image_folder)
    labels_path = Path(labels_file)

    with open(labels_path) as f:
        data = json.load(f)

    # normalize category names
    cats = {c["id"]: c["name"].lower().replace(" ", "_") for c in data["categories"]}

    # map image IDs to ground truths
    ground_truth = {}
    for a in data["annotations"]:
        img_id = a["image_id"]
        if img_id not in ground_truth:
            ground_truth[img_id] = set()
        ground_truth[img_id].add(cats.get(a["category_id"]))

    tp, fp, fn = 0, 0, 0
    hamming_scores = []

    print(f"\nBenchmarking {split_name.upper()}...")
    for img in data["images"]:
        path = image_dir / img["file_name"]
        if not path.exists():
            continue

        pred = analyze_damage(str(path))
        real = ground_truth.get(img["id"], set())

        # precision/recall
        tp += len(pred & real)
        fp += len(pred - real)
        fn += len(real - pred)

        # partial matches count for accuracy
        union = len(pred | real)
        if union == 0:
            hamming_scores.append(1.0)  # no damage
        else:
            hamming_scores.append(len(pred & real) / union)

        print(f"  {img['file_name']} | AI: {list(pred)} | Real: {list(real)}")

    # metrics
    total = len(data["images"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = sum(hamming_scores) / total

    return {"precision": precision, "recall": recall, "accuracy": accuracy}


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    root = script_dir.parent

    splits = [
        {
            "name": "train",
            "dir": root / "archive" / "train",
            "json": root / "archive" / "train" / "COCO_mul_train_annos.json",
        },
        {
            "name": "val",
            "dir": root / "archive" / "val",
            "json": root / "archive" / "val" / "COCO_mul_val_annos.json",
        },
    ]

    print("\n" + "=" * 65)
    print(f"{'SPLIT':<8} | {'PRECISION':<12} | {'RECALL':<12} | {'ACCURACY'}")
    print("-" * 65)

    
    for s in splits:
        res = run_benchmark(s["dir"], s["json"], s["name"])
        print(
            f"{s['name'].upper():<8} | {res['precision']:<12.1%} | {res['recall']:<12.1%} | {res['accuracy']:.1%}"
        )
    print("=" * 65)
