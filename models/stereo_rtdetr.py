import paddle.nn as nn
from .backbone import StereoBackbone
from .neck import StereoFusionNeck
from .head import DetectionHead

class StereoRTDETR(nn.Layer):
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=100, 
                 backbone_channels=[64, 128, 256, 512], fusion_method='concat'):
        super(StereoRTDETR, self).__init__()
        
        self.backbone = StereoBackbone(channels=backbone_channels)
        self.neck = StereoFusionNeck(
            in_channels=backbone_channels[-1], 
            hidden_dim=hidden_dim,
            fusion_method=fusion_method
        )
        self.head = DetectionHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries
        )
    
    def forward(self, left_img, right_img):
        # Extract features
        backbone_out = self.backbone(left_img, right_img)
        
        # Fuse stereo features
        fused_features = self.neck(backbone_out)
        
        # Generate predictions
        outputs = self.head(fused_features)
        
        return outputs