"""
CV Detection wrapper around the Mask R-CNN pipeline in backend/mask_rcnn/.

Model weights are loaded from paths defined in mask_rcnn/config.py:
  - models/parts_model.pth   (22-class car parts)
  - models/damage_model.pth  (9-class damage types)
"""

from mask_rcnn.inference import infer, get_device, _load_model, _load_yolo_damage_model, _load_severity_model
from mask_rcnn.config import DISPLAY_THRESHOLD


class CVDetector:
    def __init__(self):
        """Load Mask R-CNN parts model, YOLO damage model, and severity classifier."""
        self.device = get_device()
        print("Loading CV models...")
        self.parts_model = _load_model("parts", self.device)
        print("✓ Mask R-CNN (parts) loaded")
        self.yolo_damage_model = _load_yolo_damage_model()
        print("✓ YOLO (damage types) loaded")
        self.severity_model = _load_severity_model()
        if self.severity_model is not None:
            print("✓ YOLOv8-cls (severity) loaded")
        else:
            print("⚠ Severity classifier unavailable — using heuristic fallback")
        self.has_parts_model = True

    def detect(self, image_path: str, conf_threshold: float = DISPLAY_THRESHOLD) -> list[dict]:
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
            yolo_damage_model=self.yolo_damage_model,
            device=self.device,
            severity_model=self.severity_model,
        )

        detections = []
        for part_entry in result.get("damaged_parts", []):
            for dmg in part_entry.get("damage_types", []):
                if dmg["confidence"] >= conf_threshold:
                    detections.append({
                        "part":              part_entry["part"],
                        "damage_type":       dmg["type"].title(),
                        "confidence":        (part_entry["part_confidence"] + dmg["confidence"]) / 2,
                        "bbox":              dmg["damage_bbox"],
                        "part_confidence":   part_entry["part_confidence"],
                        "damage_confidence": dmg["confidence"],
                        "iou":               dmg["overlap_ratio"],
                        "severity":          dmg["severity_proxy"],
                    })

        return detections
