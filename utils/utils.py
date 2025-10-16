import os
import json
import yaml
import random
import numpy as np
import paddle
from typing import Dict, List, Any

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_directory(dir_path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)

def count_parameters(model: paddle.nn.Layer) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)

def get_class_names(num_classes: int) -> List[str]:
    """Get class names (customize based on your dataset)"""
    if num_classes == 3:
        return ['class_0', 'class_1', 'class_2']
    elif num_classes == 80:  # COCO classes
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    else:
        return [f'class_{i}' for i in range(num_classes)]

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union for two boxes [x1, y1, x2, y2]"""
    # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
    if box1.shape[-1] == 4 and len(box1.shape) == 1:
        box1 = convert_bbox_format(box1, 'center_to_corners')
    if box2.shape[-1] == 4 and len(box2.shape) == 1:
        box2 = convert_bbox_format(box2, 'center_to_corners')
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def convert_bbox_format(bbox: np.ndarray, conversion: str) -> np.ndarray:
    """Convert bounding box between different formats"""
    if conversion == 'center_to_corners':
        # [x_center, y_center, w, h] -> [x1, y1, x2, y2]
        x_center, y_center, w, h = bbox
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return np.array([x1, y1, x2, y2])
    
    elif conversion == 'corners_to_center':
        # [x1, y1, x2, y2] -> [x_center, y_center, w, h]
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([x_center, y_center, w, h])
    
    else:
        raise ValueError(f"Unknown conversion: {conversion}")

def save_results(results: List[Dict], output_path: str):
    """Save detection results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(results_path: str) -> List[Dict]:
    """Load detection results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)

def get_device() -> str:
    """Get available device (GPU or CPU)"""
    return 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'

def model_info(model: paddle.nn.Layer, verbose: bool = False):
    """Print model information"""
    total_params = count_parameters(model)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_params:,}")
    
    if verbose:
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape} - {'Trainable' if not param.stop_gradient else 'Frozen'}")