import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os

# Color palette for different classes
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64)
]

def draw_detections(image: np.ndarray, 
                   detections: List[Dict], 
                   class_names: List[str],
                   confidence_threshold: float = 0.3) -> np.ndarray:
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image (H, W, C)
        detections: List of detection dictionaries
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
    
    Returns:
        Image with drawn detections
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    for det in detections:
        if det['confidence'] < confidence_threshold:
            continue
            
        # Get bbox coordinates [x_center, y_center, width, height]
        x_center, y_center, width, height = det['bbox']
        
        # Convert to corner coordinates
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Get class info
        class_id = det['class_id']
        confidence = det['confidence']
        disparity = det.get('disparity', 0.0)
        
        # Get color for this class
        color = COLORS[class_id % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"{class_names[class_id]}: {confidence:.2f} | d: {disparity:.3f}"
        
        # Calculate text background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw text background
        cv2.rectangle(
            img, 
            (x1, y1 - text_height - baseline - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img

def visualize_stereo_detections(left_img_path: str, 
                               right_img_path: str, 
                               left_detections: List[Dict],
                               right_detections: List[Dict],
                               class_names: List[str],
                               output_path: str = None,
                               confidence_threshold: float = 0.3):
    """
    Visualize detections on stereo image pair side by side
    """
    # Load images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    # Convert BGR to RGB
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    
    # Draw detections
    left_img_with_dets = draw_detections(left_img, left_detections, class_names, confidence_threshold)
    right_img_with_dets = draw_detections(right_img, right_detections, class_names, confidence_threshold)
    
    # Concatenate images horizontally
    combined_img = np.concatenate([left_img_with_dets, right_img_with_dets], axis=1)
    
    # Add titles
    h, w = left_img_with_dets.shape[:2]
    title_height = 40
    combined_with_title = np.ones((h + title_height, w * 2, 3), dtype=np.uint8) * 255
    
    # Place images
    combined_with_title[title_height:, :w] = left_img_with_dets
    combined_with_title[title_height:, w:] = right_img_with_dets
    
    # Add text
    cv2.putText(
        combined_with_title,
        "Left Image",
        (w // 4, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )
    
    cv2.putText(
        combined_with_title,
        "Right Image",
        (w + w // 4, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )
    
    # Convert back to BGR for saving
    combined_with_title = cv2.cvtColor(combined_with_title, cv2.COLOR_RGB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, combined_with_title)
        print(f"Visualization saved to: {output_path}")
    
    return combined_with_title

def plot_training_history(train_losses: List[float], 
                         val_losses: List[float] = None,
                         output_path: str = None):
    """
    Plot training history
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {output_path}")
    
    plt.show()

def create_detection_video(image_dir: str,
                          detections_dict: Dict[str, List[Dict]],
                          class_names: List[str],
                          output_path: str,
                          fps: int = 10):
    """
    Create video from images with detections
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    if not image_files:
        print("No images found in directory")
        return
    
    # Get first image to determine size
    first_img = cv2.imread(os.path.join(image_dir, image_files[0]))
    h, w = first_img.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        
        # Get detections for this image
        detections = detections_dict.get(img_file, [])
        
        # Draw detections
        img_with_dets = draw_detections(img, detections, class_names)
        
        # Write frame
        out.write(img_with_dets)
    
    out.release()
    print(f"Video saved to: {output_path}")