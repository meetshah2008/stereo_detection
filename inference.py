import paddle
import cv2
import numpy as np
from models.stereo_rtdetr import StereoRTDETR

class StereoDetector:
    def __init__(self, model_path, num_classes=3, img_size=640):
        self.model = StereoRTDETR(num_classes=num_classes)
        self.model.eval()
        
        # Load trained weights
        checkpoint = paddle.load(model_path)
        self.model.set_state_dict(checkpoint['model_state_dict'])
        
        self.img_size = img_size
        self.num_classes = num_classes
    
    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = paddle.to_tensor(img).transpose([2, 0, 1]).unsqueeze(0)
        return img
    
    def detect(self, left_img_path, right_img_path, confidence_threshold=0.5):
        left_img = self.preprocess_image(left_img_path)
        right_img = self.preprocess_image(right_img_path)
        
        with paddle.no_grad():
            outputs = self.model(left_img, right_img)
        
        # Process outputs
        class_probs = paddle.nn.functional.softmax(outputs['class_logits'][0], axis=-1)
        bboxes = outputs['bboxes'][0]
        disparities = outputs['disparities'][0]
        
        # Filter by confidence
        max_scores = paddle.max(class_probs, axis=-1)
        keep = max_scores > confidence_threshold
        
        filtered_bboxes = bboxes[keep].numpy()
        filtered_classes = paddle.argmax(class_probs[keep], axis=-1).numpy()
        filtered_scores = max_scores[keep].numpy()
        filtered_disparities = disparities[keep].numpy()
        
        results = []
        for i in range(len(filtered_bboxes)):
            results.append({
                'bbox': filtered_bboxes[i],
                'class_id': filtered_classes[i],
                'confidence': filtered_scores[i],
                'disparity': filtered_disparities[i]
            })
        
        return results

# Usage example
if __name__ == '__main__':
    detector = StereoDetector('output/checkpoint_epoch_100.pdparams')
    
    left_path = 'path/to/left/image.jpg'
    right_path = 'path/to/right/image.jpg'
    
    results = detector.detect(left_path, right_path)
    
    for result in results:
        print(f"Class: {result['class_id']}, "
              f"Confidence: {result['confidence']:.3f}, "
              f"Disparity: {result['disparity']:.3f}, "
              f"BBox: {result['bbox']}")