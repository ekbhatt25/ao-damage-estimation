"""
Two-Model CV Detection System
- Mask R-CNN for parts detection
- Mask R-CNN for damage type detection
- IoU-based merging
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms as T
import numpy as np

class CVDetector:
    def __init__(self, 
                 parts_model_path='models/parts_model.pth',
                 damage_model_path='models/damage_model.pth'):
        """
        Initialize two-model detection system
        
        Args:
            parts_model_path: Path to Mask R-CNN parts weights
            damage_model_path: Path to Mask R-CNN damage weights
        """
        
        print("Loading CV models...")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load parts model
        self.parts_model = self._load_model(parts_model_path, num_classes=21)  # Adjust num_classes
        print("✓ Mask R-CNN (parts) loaded")
        
        # Load damage model
        self.damage_model = self._load_model(damage_model_path, num_classes=6)  # Adjust num_classes
        print("✓ Mask R-CNN (damage types) loaded")
        
        # Transform
        self.transform = T.Compose([T.ToTensor()])
        
        # IoU threshold
        self.iou_threshold = 0.3
        
        # Class names (update these based on your training)
        self.part_names = {
            1: 'Back-bumper',
            2: 'Back-door',
            3: 'Back-wheel',
            4: 'Back-window',
            5: 'Back-windshield',
            6: 'Fender',
            7: 'Front-bumper',
            8: 'Front-door',
            8: 'Front-wheel',
            9: 'Front-window',
            10: 'Grille',
            11: 'Headlight',
            12: 'Hood',
            13: 'License-plate',
            14: 'Quarter-panel',
            15: 'Rocker-panel',
            16: 'Roof',
            17: 'Tail-light',
            18: 'Trunk',
            19: 'Windshield'          
        }
        
        self.damage_names = {
            1: 'dent',
            2: 'scratch',
            3: 'crack',
            4: 'glass_shatter',
            5: 'lamp_broken',
            6: 'tire_flat',
        }
    
    def _load_model(self, model_path, num_classes):
        """Load Mask R-CNN model"""
        
        # Create model
        model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def detect(self, image_path, conf_threshold=0.25):
        """
        Run two-model detection and merge results
        
        Args:
            image_path: Path to image
            conf_threshold: Minimum confidence threshold
        
        Returns:
            List of detections with part, damage_type, bbox, confidence
        """
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).to(self.device)
        
        # STEP 1: Detect parts
        with torch.no_grad():
            parts_pred = self.parts_model([image_tensor])[0]
        
        parts = self._extract_detections(
            parts_pred,
            self.part_names,
            conf_threshold
        )
        
        # STEP 2: Detect damage types
        with torch.no_grad():
            damage_pred = self.damage_model([image_tensor])[0]
        
        damages = self._extract_detections(
            damage_pred,
            self.damage_names,
            conf_threshold
        )
        
        # STEP 3: Merge based on IoU
        merged = self._merge_detections(parts, damages)
        
        return merged
    
    def _extract_detections(self, prediction, class_names, conf_threshold):
        """Extract detections from Mask R-CNN output"""
        
        detections = []
        
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= conf_threshold:
                detections.append({
                    'label': class_names.get(labels[i], f'class_{labels[i]}'),
                    'bbox': boxes[i].tolist(),
                    'confidence': float(scores[i])
                })
        
        return detections
    
    def _merge_detections(self, parts, damages):
        """
        Merge part and damage detections using IoU
        
        For each damage detection, find best matching part
        """
        
        merged = []
        
        for damage in damages:
            best_match = None
            best_iou = 0
            
            # Find best matching part
            for part in parts:
                iou = self._calculate_iou(damage['bbox'], part['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = part
            
            # Create merged detection
            if best_match:
                merged.append({
                    'part': best_match['label'],
                    'damage_type': damage['label'],
                    'bbox': damage['bbox'],
                    'confidence': (damage['confidence'] + best_match['confidence']) / 2,
                    'part_confidence': best_match['confidence'],
                    'damage_confidence': damage['confidence'],
                    'iou': best_iou
                })
            else:
                # No matching part
                merged.append({
                    'part': 'unknown_part',
                    'damage_type': damage['label'],
                    'bbox': damage['bbox'],
                    'confidence': damage['confidence'],
                    'part_confidence': 0.0,
                    'damage_confidence': damage['confidence'],
                    'iou': 0.0
                })
        
        return merged
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0.0
        
        return iou


if __name__ == "__main__":
    # Test
    detector = CVDetector()
    
    results = detector.detect('test_car.jpg')
    
    print("\nDetection Results:")
    print("="*60)
    for det in results:
        print(f"Part: {det['part']}")
        print(f"Damage: {det['damage_type']}")
        print(f"Confidence: {det['confidence']:.2f}")
        print(f"IoU: {det['iou']:.2f}")
        print("-"*60)