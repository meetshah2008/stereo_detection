import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class StereoFusionNeck(nn.Layer):
    def __init__(self, in_channels=512, hidden_dim=256, fusion_method='concat'):
        super(StereoFusionNeck, self).__init__()
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim
        
        # Fusion convolution
        if fusion_method == 'concat':
            self.fusion_conv = nn.Sequential(
                nn.Conv2D(in_channels * 2, hidden_dim, 1),
                nn.BatchNorm2D(hidden_dim),
                nn.ReLU(),
                nn.Conv2D(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2D(hidden_dim),
                nn.ReLU()
            )
        elif fusion_method == 'add':
            self.fusion_conv = nn.Sequential(
                nn.Conv2D(in_channels, hidden_dim, 1),
                nn.BatchNorm2D(hidden_dim),
                nn.ReLU()
            )
    
    def forward(self, backbone_output):
        left_features = backbone_output['left_features']
        right_features = backbone_output['right_features']
        
        # Use the last feature map for fusion (you can extend this to multi-scale)
        left_feat = left_features[-1]
        right_feat = right_features[-1]
        
        if self.fusion_method == 'concat':
            fused = paddle.concat([left_feat, right_feat], axis=1)
            fused = self.fusion_conv(fused)
        elif self.fusion_method == 'add':
            left_proj = self.fusion_conv(left_feat)
            right_proj = self.fusion_conv(right_feat)
            fused = left_proj + right_proj
        else:
            fused = left_feat  # Fallback to left features only
        
        return fused