import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class StereoDetectionLoss(nn.Layer):
    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, disparity_weight=0.1):
        super(StereoDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.disparity_weight = disparity_weight
        
    def forward(self, predictions, targets):
        class_logits = predictions['class_logits']  # [B, N, C]
        pred_bboxes = predictions['bboxes']  # [B, N, 4]
        pred_disparities = predictions['disparities']  # [B, N]
        
        gt_bboxes = targets['bboxes']  # List of [M, 4]
        gt_classes = targets['classes']  # List of [M]
        gt_disparities = targets['disparities']  # [B, M]
        
        batch_size = class_logits.shape[0]
        total_loss = 0
        
        for i in range(batch_size):
            # Classification loss (Focal Loss)
            cls_loss = self.focal_loss(class_logits[i], gt_classes[i])
            
            # Bounding box loss (L1 + GIoU)
            bbox_loss = self.bbox_loss(pred_bboxes[i], gt_bboxes[i])
            
            # Disparity loss (only for matched objects)
            disparity_loss = self.disparity_loss(pred_disparities[i], gt_disparities[i])
            
            total_loss += cls_loss + bbox_loss + self.disparity_weight * disparity_loss
        
        return total_loss / batch_size
    
    def focal_loss(self, pred, target):
        # Simple cross entropy for now (you can implement proper focal loss)
        if len(target) == 0:
            return paddle.to_tensor(0.0)
        
        # For simplicity, assign each GT to first N queries
        num_objects = min(pred.shape[0], len(target))
        pred = pred[:num_objects]
        target = target[:num_objects]
        
        return F.cross_entropy(pred, target)
    
    def bbox_loss(self, pred, target):
        if len(target) == 0:
            return paddle.to_tensor(0.0)
        
        num_objects = min(pred.shape[0], len(target))
        pred = pred[:num_objects]
        target = target[:num_objects]
        
        l1_loss = F.l1_loss(pred, target)
        return l1_loss
    
    def disparity_loss(self, pred, target):
        if len(target) == 0:
            return paddle.to_tensor(0.0)
        
        num_objects = min(pred.shape[0], len(target))
        pred = pred[:num_objects]
        target = target[:num_objects]
        
        return F.mse_loss(pred, target)