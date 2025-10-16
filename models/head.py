import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DetectionHead(nn.Layer):
    def __init__(self, hidden_dim=256, num_classes=80, num_queries=100):
        super(DetectionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Bounding box head (x, y, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # Normalize to [0,1]
        )
        
        # Disparity head
        self.disparity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0,1]
        )
    
    def forward(self, fused_features):
        # Global average pooling
        B, C, H, W = fused_features.shape
        features_pooled = F.adaptive_avg_pool2d(fused_features, 1).squeeze(-1).squeeze(-1)  # [B, C]
        
        # Expand features for queries
        features_expanded = features_pooled.unsqueeze(1).expand([B, self.num_queries, C])  # [B, N, C]
        
        # Add query embeddings
        query_emb = self.query_embed.weight.unsqueeze(0).expand([B, self.num_queries, C])
        features_with_queries = features_expanded + query_emb
        
        # Predictions
        class_logits = self.class_head(features_with_queries)  # [B, N, num_classes]
        bboxes = self.bbox_head(features_with_queries)  # [B, N, 4]
        disparities = self.disparity_head(features_with_queries)  # [B, N, 1]
        
        return {
            'class_logits': class_logits,
            'bboxes': bboxes,
            'disparities': disparities.squeeze(-1)  # [B, N]
        }