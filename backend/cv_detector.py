"""
CV Detection wrapper around the Mask R-CNN pipeline in backend/mask_rcnn/.

Model weights are loaded from paths defined in mask_rcnn/config.py:
  - models/parts_model.pth   (22-class car parts)
  - models/damage_model.pth  (9-class damage types)
"""

from mask_rcnn.inference import infer, get_device, _load_model


class CVDetector:
    def __init__(self):
        """Load both Mask R-CNN models. Paths come from mask_rcnn/config.py."""
        self.device = get_device()
        print("Loading CV models...")
        self.parts_model = _load_model("parts", self.device)
        print("✓ Mask R-CNN (parts) loaded")
        self.damage_model = _load_model("damage", self.device)
        print("✓ Mask R-CNN (damage types) loaded")
        self.has_parts_model = True

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> list[dict]:
        """
        Run the two-model Mask R-CNN pipeline on an image.

        Returns a flat list of detections, one entry per (part, damage_type) pair:
            {
                part, damage_type, confidence, bbox,
                part_confidence, damage_confidence, iou, severity
            }
        """
        result = infer(
            image_path,
            parts_model=self.parts_model,
            damage_model=self.damage_model,
            device=self.device,
        )

        detections = []
        for part_entry in result.get("damaged_parts", []):
            for dmg in part_entry.get("damage_types", []):
                if dmg["confidence"] >= conf_threshold:
                    detections.append({
                        "part":              part_entry["part"],
                        "damage_type":       dmg["type"],
                        "confidence":        (part_entry["part_confidence"] + dmg["confidence"]) / 2,
                        "bbox":              dmg["damage_bbox"],
                        "part_confidence":   part_entry["part_confidence"],
                        "damage_confidence": dmg["confidence"],
                        "iou":               dmg["overlap_ratio"],
                        "severity":          dmg["severity_proxy"],
                    })

        return detections
