"""
Mask R-CNN model factory.

Uses torchvision's maskrcnn_resnet50_fpn pretrained on COCO as the backbone.
The box and mask predictor heads are replaced to match our number of classes.

Two models are created independently:
  - parts_model:  21 car-part classes + background
  - damage_model: 8 damage-type classes + background
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as grad_ckpt
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.resnet import Bottleneck

from .config import NUM_PART_CLASSES, NUM_DAMAGE_CLASSES


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Patch every ResNet Bottleneck block in the backbone to use activation
    checkpointing, trading compute for memory during backprop.
    """
    for module in model.backbone.body.modules():
        if isinstance(module, Bottleneck):
            _orig = module.forward
            module.forward = (
                lambda *args, fn=_orig:
                grad_ckpt.checkpoint(fn, *args, use_reentrant=False)
            )


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a Mask R-CNN with a ResNet-50-FPN backbone pretrained on COCO.
    The final box and mask heads are replaced for `num_classes`.
    Image resize is capped at 640/800 (down from 800/1333) to reduce
    activation memory on the Jetson Orin Nano.

    Args:
        num_classes: total number of classes INCLUDING background (index 0)
        pretrained:  load COCO-pretrained backbone weights
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        min_size=640,
        max_size=800,
    )

    # Replace box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def build_parts_model(pretrained: bool = True) -> nn.Module:
    return build_model(NUM_PART_CLASSES, pretrained)


def build_damage_model(pretrained: bool = True) -> nn.Module:
    return build_model(NUM_DAMAGE_CLASSES, pretrained)


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze backbone + FPN so only the RPN and ROI heads are trained.
    Use during Phase 1 (head warm-up) to prevent destroying pretrained features.
    """
    for name, param in model.named_parameters():
        if "backbone" in name or "fpn" in name:
            param.requires_grad_(False)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning in Phase 2."""
    for param in model.parameters():
        param.requires_grad_(True)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """Load a saved checkpoint and return the metadata dict."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def count_params(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
