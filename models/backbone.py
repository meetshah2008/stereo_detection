import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, 3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class StereoBackbone(nn.Layer):
    def __init__(self, num_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], share_weights=True):
        super(StereoBackbone, self).__init__()
        self.share_weights = share_weights
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2D(3, 64, 7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.pool = nn.MaxPool2D(3, stride=2, padding=1)
        
        # Left image backbone
        self.left_layers = self._make_layers(channels, num_blocks)
        
        # Right image backbone
        if share_weights:
            self.right_layers = self.left_layers
        else:
            self.right_layers = self._make_layers(channels, num_blocks)
    
    def _make_layers(self, channels, num_blocks):
        layers = []
        current_channels = self.in_channels
        
        for i, (out_channels, num_block) in enumerate(zip(channels, num_blocks)):
            stride = 1 if i == 0 else 2
            layers.append(self._make_layer(current_channels, out_channels, num_block, stride))
            current_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, left_img, right_img):
        # Process left image
        left_feat = F.relu(self.bn1(self.conv1(left_img)))
        left_feat = self.pool(left_feat)
        left_features = []
        for layer in self.left_layers:
            left_feat = layer(left_feat)
            left_features.append(left_feat)
        
        # Process right image
        right_feat = F.relu(self.bn1(self.conv1(right_img)))
        right_feat = self.pool(right_feat)
        right_features = []
        for layer in self.right_layers:
            right_feat = layer(right_feat)
            right_features.append(right_feat)
        
        return {
            'left_features': left_features,
            'right_features': right_features
        }