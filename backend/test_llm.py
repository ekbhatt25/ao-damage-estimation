import json

def load_coco_data():
    """Load and parse COCO dataset"""
    with open('../archive/train/COCO_mul_train_annos.json') as f:
        coco_data = json.load(f)
    
    # parse COCO format
    images_list = coco_data['images']
    annotations_list = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # group annotations by image
    image_annotations = {}
    for anno in annotations_list:
        img_id = anno['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(categories[anno['category_id']])
    
    # create test cases
    test_cases = []
    for img in images_list:
        parts = image_annotations.get(img['id'], [])
        if parts:
            test_cases.append({
                'image_id': img['id'],
                'filename': img['file_name'],
                'damaged_parts': list(set(parts))  # Remove duplicates
            })
    
    return test_cases


def generate_prompt(damaged_parts_ground_truth):
    """  
    Arguments:
        damaged_parts_ground_truth: list like ["front_bumper", "door"]
    
    Returns:
        prompt (str): The prompt to send to LLM
        mock_data (dict): The mock data used (for validation later)
    """
    
    # create mock data
    mock_data = create_mock_data(damaged_parts_ground_truth)
    
    # format damaged parts for prompt
    parts_list = []
    for part in mock_data["damaged_parts"]:
        parts_list.append(
            f"- {part['part']}: {part['severity']} {part['damage_type']} "
            f"(recommended: {part['action']}, estimated cost: ${part['cost_range'][0]}-${part['cost_range'][1]})"
        )
    
    parts_formatted = "\n".join(parts_list)
    
    # build prompt
    prompt = f"""You are an insurance adjuster assistant analyzing vehicle damage.

Vehicle: {mock_data['vehicle']['year']} {mock_data['vehicle']['make']} {mock_data['vehicle']['model']}
Location: ZIP {mock_data['zip_code']}

Damage Assessment from CV and Cost Models:

{parts_formatted}

Total Estimated Cost: ${mock_data['total_cost_range'][0]}-${mock_data['total_cost_range'][1]}

Generate a professional insurance claim assessment.

Return your response in this exact JSON format:
{{
  "damaged_parts": [
    {{
      "part": "part_name",
      "damage_type": "type",
      "severity": "severity_level",
      "action": "repair_or_replace",
      "cost_range": [low, high]
    }}
  ],
  "total_cost_range": [low, high],
  "explanation": "Professional damage assessment (2-4 sentences)",
  "confidence_score": 0.85,
  "stp_eligible": true,
  "stp_reasoning": "Reasoning for STP decision"
}}

Guidelines:
- explanation: Reference specific damage, justify repair decisions and costs
- confidence_score (0.0-1.0): 
  * 0.9-1.0 = Very confident, standard damage
  * 0.7-0.9 = Confident, typical case
  * 0.5-0.7 = Moderate confidence
  * <0.5 = Low confidence
- stp_eligible: true if straightforward and cost <$1500, false otherwise
- stp_reasoning: One sentence explaining STP decision

Return ONLY valid JSON with no other text."""

    return prompt, mock_data


def create_mock_data(damaged_parts_ground_truth):
    """Generate realistic mock CV + Cost model outputs"""
    
    import random
    
    # fixed vehicle for testing
    mock_data = {
        "vehicle": {
            "year": 2019,
            "make": "Honda",
            "model": "Civic"
        },
        "zip_code": "48823",
        "damaged_parts": []
    }
    
    # cost estimates
    part_costs = {
        "front_bumper": {"repair": 500, "replace": 850},
        "rear_bumper": {"repair": 500, "replace": 850},
        "door": {"repair": 450, "replace": 1200},
        "hood": {"repair": 600, "replace": 1500},
        "headlamp": {"repair": 200, "replace": 500}
    }
    
    # damage types
    damage_types_map = {
        "front_bumper": ["crack", "dent", "scratch"],
        "rear_bumper": ["crack", "dent", "scratch"],
        "door": ["dent", "scratch"],
        "hood": ["dent", "scratch"],
        "headlamp": ["crack", "broken"]
    }
    
    for part in damaged_parts_ground_truth:
        # random severity
        severity = random.choice(["minor", "moderate", "major"])
        
        # damage type based on part
        damage_type = random.choice(damage_types_map.get(part, ["damage"]))
        
        # action based on severity
        if severity == "major":
            action = "replace"
        elif severity == "moderate":
            action = random.choice(["repair", "replace"])
        else:
            action = "repair"
        
        # get base cost
        base_cost = part_costs.get(part, {}).get(action, 500)
        
        # cost range (±20%)
        cost_low = int(base_cost * 0.8)
        cost_high = int(base_cost * 1.2)
        
        mock_data["damaged_parts"].append({
            "part": part,
            "damage_type": damage_type,
            "severity": severity,
            "action": action,
            "cost_range": [cost_low, cost_high],
            "estimated_cost": base_cost
        })
    
    # total cost range
    total_low = sum(p["cost_range"][0] for p in mock_data["damaged_parts"])
    total_high = sum(p["cost_range"][1] for p in mock_data["damaged_parts"])
    mock_data["total_cost_range"] = [total_low, total_high]
    
    return mock_data