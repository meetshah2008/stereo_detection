import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset
import random

class StereoDetectionDataset(Dataset):
    def __init__(self, data_root, img_size=640, augment=True):
        """
        Args:
            data_root: Root directory containing 'left', 'right', and 'txt' folders
            img_size: Input image size
            augment: Whether to apply data augmentation
        """
        self.data_root = data_root
        self.img_size = img_size
        self.augment = augment
        
        # Get all image pairs
        self.left_dir = os.path.join(data_root, 'left')
        self.right_dir = os.path.join(data_root, 'right') 
        self.txt_dir = os.path.join(data_root, 'txt')
        
        # Get all image names (assuming same names in left/right folders)
        self.image_names = [f for f in os.listdir(self.left_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(self.image_names)} stereo image pairs")
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Load stereo images
        left_img = cv2.imread(os.path.join(self.left_dir, img_name))
        right_img = cv2.imread(os.path.join(self.right_dir, img_name))
        
        # Load labels
        txt_path = os.path.join(self.txt_dir, base_name + '.txt')
        labels = self.load_labels(txt_path)
        
        # Data augmentation
        if self.augment:
            left_img, right_img, labels = self.augment_images(left_img, right_img, labels)
        
        # Preprocess images
        left_img = self.preprocess_image(left_img)
        right_img = self.preprocess_image(right_img)
        
        # Convert to tensor
        left_img = paddle.to_tensor(left_img).transpose([2, 0, 1]).astype('float32')
        right_img = paddle.to_tensor(right_img).transpose([2, 0, 1]).astype('float32')
        
        # Prepare labels
        bboxes = []
        classes = []
        disparities = []
        
        for label in labels:
            class_id, x_center, y_center, width, height, disparity = label
            bboxes.append([x_center, y_center, width, height])
            classes.append(class_id)
            disparities.append(disparity)
        
        return {
            'left_image': left_img,
            'right_image': right_img,
            'bboxes': paddle.to_tensor(bboxes).astype('float32'),
            'classes': paddle.to_tensor(classes).astype('int64'),
            'disparities': paddle.to_tensor(disparities).astype('float32'),
            'img_name': img_name
        }
    
    def load_labels(self, txt_path):
        """Load labels from YOLO format text file with disparity"""
        labels = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 6:  # class_id, x, y, w, h, disparity
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        disparity = float(parts[5])
                        labels.append([class_id, x_center, y_center, width, height, disparity])
        return labels
    
    def preprocess_image(self, img):
        """Resize and normalize image"""
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        return img
    
    def augment_images(self, left_img, right_img, labels):
        """Simple data augmentation for stereo images"""
        # Color jitter
        if random.random() < 0.5:
            # Apply same color jitter to both images
            brightness = random.uniform(0.8, 1.2)
            left_img = cv2.convertScaleAbs(left_img, alpha=brightness, beta=0)
            right_img = cv2.convertScaleAbs(right_img, alpha=brightness, beta=0)
        
        # Horizontal flip (affects disparity!)
        if random.random() < 0.5:
            left_img = cv2.flip(left_img, 1)
            right_img = cv2.flip(right_img, 1)
            # Flip bbox coordinates
            for label in labels:
                label[1] = 1.0 - label[1]  # Flip x_center
                # Note: Disparity might need adjustment based on flip
        
        return left_img, right_img, labels