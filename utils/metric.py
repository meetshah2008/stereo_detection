import numpy as np
from typing import List, Dict, Tuple
from .utils import calculate_iou, convert_bbox_format

class DetectionMetrics:
    """Calculate detection metrics like mAP, precision, recall"""
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
        self.gt_count = [0] * self.num_classes
    
    def update(self, 
               pred_bboxes: List[np.ndarray], 
               pred_classes: List[int],
               pred_scores: List[float],
               gt_bboxes: List[np.ndarray],
               gt_classes: List[int]):
        """
        Update metrics with new batch of predictions and ground truth
        
        Args:
            pred_bboxes: List of predicted bounding boxes [x_center, y_center, w, h]
            pred_classes: List of predicted class IDs
            pred_scores: List of prediction confidence scores
            gt_bboxes: List of ground truth bounding boxes [x_center, y_center, w, h]
            gt_classes: List of ground truth class IDs
        """
        # Convert bboxes to corner format for IoU calculation
        pred_bboxes_corners = [convert_bbox_format(bbox, 'center_to_corners') for bbox in pred_bboxes]
        gt_bboxes_corners = [convert_bbox_format(bbox, 'center_to_corners') for bbox in gt_bboxes]
        
        # Update ground truth counts
        for class_id in gt_classes:
            if class_id < self.num_classes:
                self.gt_count[class_id] += 1
        
        # Match predictions to ground truth
        used_gt = [False] * len(gt_bboxes)
        
        for pred_idx, (pred_bbox, pred_class, pred_score) in enumerate(zip(pred_bboxes_corners, pred_classes, pred_scores)):
            if pred_class >= self.num_classes:
                continue
                
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, (gt_bbox, gt_class) in enumerate(zip(gt_bboxes_corners, gt_classes)):
                if used_gt[gt_idx] or gt_class != pred_class:
                    continue
                    
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if prediction is true positive or false positive
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                self.tp[pred_class] += 1
                used_gt[best_gt_idx] = True
            else:
                self.fp[pred_class] += 1
        
        # Count false negatives (unmatched ground truths)
        for gt_idx, (used, gt_class) in enumerate(zip(used_gt, gt_classes)):
            if not used and gt_class < self.num_classes:
                self.fn[gt_class] += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        
        for class_id in range(self.num_classes):
            tp = self.tp[class_id]
            fp = self.fp[class_id]
            fn = self.fn[class_id]
            gt = self.gt_count[class_id]
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1-score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'class_{class_id}_precision'] = precision
            metrics[f'class_{class_id}_recall'] = recall
            metrics[f'class_{class_id}_f1'] = f1
        
        # Mean metrics
        precisions = [metrics[f'class_{i}_precision'] for i in range(self.num_classes)]
        recalls = [metrics[f'class_{i}_recall'] for i in range(self.num_classes)]
        f1_scores = [metrics[f'class_{i}_f1'] for i in range(self.num_classes)]
        
        metrics['mean_precision'] = np.mean(precisions)
        metrics['mean_recall'] = np.mean(recalls)
        metrics['mean_f1'] = np.mean(f1_scores)
        metrics['map'] = metrics['mean_precision']  # Simple mAP approximation
        
        return metrics

class DisparityMetrics:
    """Calculate disparity estimation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset disparity metrics"""
        self.errors = []
        self.abs_errors = []
    
    def update(self, pred_disparities: List[float], gt_disparities: List[float]):
        """Update disparity metrics"""
        for pred, gt in zip(pred_disparities, gt_disparities):
            error = pred - gt
            abs_error = abs(error)
            
            self.errors.append(error)
            self.abs_errors.append(abs_error)
    
    def compute(self) -> Dict[str, float]:
        """Compute disparity metrics"""
        if not self.errors:
            return {}
        
        errors = np.array(self.errors)
        abs_errors = np.array(self.abs_errors)
        
        return {
            'disparity_mae': np.mean(abs_errors),
            'disparity_mse': np.mean(errors ** 2),
            'disparity_rmse': np.sqrt(np.mean(errors ** 2)),
            'disparity_mean_error': np.mean(errors),
            'disparity_std_error': np.std(errors)
        }

def evaluate_model(model, dataloader, num_classes, iou_threshold=0.5):
    """
    Comprehensive model evaluation
    """
    detection_metrics = DetectionMetrics(num_classes, iou_threshold)
    disparity_metrics = DisparityMetrics()
    
    model.eval()
    
    for batch in dataloader:
        # Get predictions (you'll need to adapt this based on your model output)
        left_imgs = batch['left_image']
        right_imgs = batch['right_image']
        
        # Forward pass
        outputs = model(left_imgs, right_imgs)
        
        # Process outputs (this depends on your model output format)
        # You'll need to extract predictions and convert to the expected format
        
        # For now, this is a placeholder - you'll need to implement based on your model
        pass
    
    detection_results = detection_metrics.compute()
    disparity_results = disparity_metrics.compute()
    
    # Combine results
    results = {**detection_results, **disparity_results}
    
    return results