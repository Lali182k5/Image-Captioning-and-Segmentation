"""
Alternative Image Segmentation Module using YOLOv8
This module provides functionality for object detection and instance segmentation using YOLOv8.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import logging
from typing import Tuple, List, Dict, Union, Optional
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Please install ultralytics for segmentation functionality.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO class names for object detection (same as original Detectron2)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class YOLOSegmenter:
    """
    A class for performing object detection and instance segmentation using YOLOv8.
    """
    
    def __init__(self, 
                 model_name: str = "yolov8n-seg.pt",
                 score_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Initialize the YOLOSegmenter with the specified model.
        
        Args:
            model_name (str): YOLO model name (yolov8n-seg.pt, yolov8s-seg.pt, etc.)
            score_threshold (float): Minimum confidence score for detections
            device (str): Device to run inference on ("auto", "cpu", or "cuda")
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not installed. Please install ultralytics to use segmentation features.")
        
        self.model_name = model_name
        self.score_threshold = score_threshold
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup the YOLO model."""
        try:
            logger.info(f"Setting up YOLO model: {self.model_name}")
            
            # Load YOLO model
            self.model = YOLO(self.model_name)
            
            # Set device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
            else:
                self.model.to("cpu")
            
            logger.info("YOLO model setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up YOLO model: {e}")
            raise
    
    def segment_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Perform instance segmentation on the given image.
        
        Args:
            image_input (Union[str, np.ndarray, Image.Image]): Image path, numpy array, or PIL Image
            
        Returns:
            Tuple containing:
            - masks (np.ndarray): Binary masks for detected objects
            - classes (np.ndarray): Class IDs for detected objects
            - scores (np.ndarray): Confidence scores for detections
            - class_names (List[str]): Human-readable class names
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img_path = image_input
            elif isinstance(image_input, Image.Image):
                # Save PIL image temporarily
                img_path = "temp_yolo_image.jpg"
                image_input.save(img_path)
            elif isinstance(image_input, np.ndarray):
                # Save numpy array as image
                img_path = "temp_yolo_image.jpg"
                if image_input.dtype != np.uint8:
                    image_input = (image_input * 255).astype(np.uint8)
                cv2.imwrite(img_path, image_input)
            else:
                raise ValueError("image_input must be a file path, numpy array, or PIL Image")
            
            # Perform inference
            results = self.model(img_path, conf=self.score_threshold, verbose=False)
            
            # Extract results from the first (and only) image
            result = results[0]
            
            # Initialize empty arrays
            masks = np.array([])
            classes = np.array([])
            scores = np.array([])
            class_names = []
            
            # Check if we have segmentation masks
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # Shape: (N, H, W)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                
                # Get class names
                class_names = [COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}" 
                              for class_id in classes]
            elif result.boxes is not None:
                # If no masks but have boxes, create dummy masks
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                
                # Create rectangular masks from bounding boxes
                img_height, img_width = result.orig_shape
                masks = np.zeros((len(boxes), img_height, img_width), dtype=bool)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    masks[i, y1:y2, x1:x2] = True
                
                class_names = [COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}" 
                              for class_id in classes]
            
            # Clean up temporary files
            if isinstance(image_input, (Image.Image, np.ndarray)) and os.path.exists(img_path):
                os.remove(img_path)
            
            logger.info(f"YOLO detected {len(classes)} objects")
            return masks, classes, scores, class_names
            
        except Exception as e:
            logger.error(f"Error in YOLO segmentation: {e}")
            return np.array([]), np.array([]), np.array([]), []
    
    def visualize_predictions(self, 
                            image_input: Union[str, np.ndarray, Image.Image],
                            show_boxes: bool = True,
                            show_masks: bool = True,
                            show_labels: bool = True) -> np.ndarray:
        """
        Visualize segmentation results on the image using YOLO's built-in visualization.
        
        Args:
            image_input (Union[str, np.ndarray, Image.Image]): Image path, numpy array, or PIL Image
            show_boxes (bool): Whether to show bounding boxes
            show_masks (bool): Whether to show segmentation masks
            show_labels (bool): Whether to show class labels
            
        Returns:
            np.ndarray: Image with visualizations overlaid
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img_path = image_input
                original_img = cv2.imread(image_input)
            elif isinstance(image_input, Image.Image):
                img_path = "temp_yolo_viz.jpg"
                image_input.save(img_path)
                original_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                img_path = "temp_yolo_viz.jpg"
                if image_input.dtype != np.uint8:
                    image_input = (image_input * 255).astype(np.uint8)
                cv2.imwrite(img_path, image_input)
                original_img = image_input
            else:
                raise ValueError("image_input must be a file path, numpy array, or PIL Image")
            
            # Perform inference
            results = self.model(img_path, conf=self.score_threshold, verbose=False)
            
            # Get annotated image
            annotated_img = results[0].plot(
                boxes=show_boxes,
                masks=show_masks,
                labels=show_labels,
                conf=True
            )
            
            # Clean up temporary files
            if isinstance(image_input, (Image.Image, np.ndarray)) and os.path.exists(img_path):
                os.remove(img_path)
            
            return annotated_img
            
        except Exception as e:
            logger.error(f"Error in YOLO visualization: {e}")
            # Return original image if visualization fails
            if isinstance(image_input, str):
                return cv2.imread(image_input)
            elif isinstance(image_input, Image.Image):
                return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                return image_input
    
    def get_detection_summary(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Get a summary of detected objects in the image.
        
        Args:
            image_input (Union[str, np.ndarray, Image.Image]): Image path, numpy array, or PIL Image
            
        Returns:
            Dict: Summary containing object counts and details
        """
        try:
            masks, classes, scores, class_names = self.segment_image(image_input)
            
            if len(classes) == 0:
                return {"total_objects": 0, "object_counts": {}, "detections": []}
            
            # Count objects by class
            object_counts = {}
            detections = []
            
            for i, (class_id, class_name, score) in enumerate(zip(classes, class_names, scores)):
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                mask_area = int(np.sum(masks[i])) if len(masks) > i else 0
                detections.append({
                    "object_id": i,
                    "class_name": class_name,
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "mask_area": mask_area
                })
            
            summary = {
                "total_objects": len(classes),
                "object_counts": object_counts,
                "detections": detections,
                "average_confidence": float(np.mean(scores))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating YOLO detection summary: {e}")
            return {"total_objects": 0, "object_counts": {}, "detections": [], "error": str(e)}

# Initialize global YOLO segmenter instance
_yolo_segmenter = None

def get_yolo_segmenter() -> Optional[YOLOSegmenter]:
    """
    Get or create a global YOLOSegmenter instance.
    
    Returns:
        YOLOSegmenter: Global segmenter instance, or None if YOLO is not available
    """
    global _yolo_segmenter
    if not YOLO_AVAILABLE:
        return None
    
    if _yolo_segmenter is None:
        try:
            _yolo_segmenter = YOLOSegmenter()
        except Exception as e:
            logger.error(f"Failed to initialize YOLO segmenter: {e}")
            return None
    return _yolo_segmenter

# Update the original functions to use YOLO instead of Detectron2
def get_segmenter():
    """Get YOLO segmenter instead of Detectron2."""
    return get_yolo_segmenter()

def segment_image(image_path: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to segment an image using YOLO.
    
    Args:
        image_path (Union[str, np.ndarray, Image.Image]): Image path, numpy array, or PIL Image
        
    Returns:
        Tuple containing masks and classes
    """
    if not YOLO_AVAILABLE:
        logger.warning("YOLO not available. Segmentation functionality is disabled.")
        return np.array([]), np.array([])
    
    segmenter = get_yolo_segmenter()
    if segmenter is None:
        logger.error("Could not initialize YOLO segmenter")
        return np.array([]), np.array([])
    
    masks, classes, _, _ = segmenter.segment_image(image_path)
    return masks, classes

def visualize_segmentation(image_path: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convenience function to visualize segmentation results using YOLO.
    
    Args:
        image_path (Union[str, np.ndarray, Image.Image]): Image path, numpy array, or PIL Image
        
    Returns:
        np.ndarray: Image with visualizations
    """
    if not YOLO_AVAILABLE:
        logger.error("YOLO not available for segmentation")
        if isinstance(image_path, str):
            return cv2.imread(image_path)
        return image_path
    
    segmenter = get_yolo_segmenter()
    if segmenter is None:
        logger.error("Could not initialize YOLO segmenter")
        if isinstance(image_path, str):
            return cv2.imread(image_path)
        return image_path
    
    return segmenter.visualize_predictions(image_path)

if __name__ == "__main__":
    # Test the YOLO segmentation functionality
    import sys
    
    if not YOLO_AVAILABLE:
        print("YOLO is not available. Please install ultralytics to test segmentation.")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            masks, classes = segment_image(image_path)
            print(f"YOLO detected {len(classes)} objects")
            
            # Get detailed summary
            segmenter = get_yolo_segmenter()
            if segmenter:
                summary = segmenter.get_detection_summary(image_path)
                print("Detection Summary:")
                print(f"Total Objects: {summary['total_objects']}")
                print(f"Object Counts: {summary['object_counts']}")
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python yolo_segmentation.py <image_path>")