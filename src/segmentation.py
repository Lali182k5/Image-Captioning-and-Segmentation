"""
Image Segmentation Module using Detectron2
This module provides functionality for object detection and instance segmentation.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import logging
from typing import Tuple, List, Dict, Union, Optional
import os

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    logging.warning("Detectron2 not available. Please install detectron2 for segmentation functionality.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO class names for object detection
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

class ImageSegmenter:
    """
    A class for performing object detection and instance segmentation using Detectron2.
    """
    
    def __init__(self, 
                 model_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                 score_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Initialize the ImageSegmenter with the specified model.
        
        Args:
            model_config (str): Model configuration from Detectron2 model zoo
            score_threshold (float): Minimum confidence score for detections
            device (str): Device to run inference on ("auto", "cpu", or "cuda")
        """
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 is not installed. Please install detectron2 to use segmentation features.")
        
        self.model_config = model_config
        self.score_threshold = score_threshold
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.predictor = None
        self.cfg = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup the Detectron2 model configuration and predictor."""
        try:
            logger.info(f"Setting up Detectron2 model: {self.model_config}")
            
            # Create configuration
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(self.model_config))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_config)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
            self.cfg.MODEL.DEVICE = self.device
            
            # Create predictor
            self.predictor = DefaultPredictor(self.cfg)
            logger.info("Model setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
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
                img = cv2.imread(image_input)
                if img is None:
                    raise ValueError(f"Could not load image: {image_input}")
            elif isinstance(image_input, Image.Image):
                img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                img = image_input
            else:
                raise ValueError("image_input must be a file path, numpy array, or PIL Image")
            
            # Perform inference
            outputs = self.predictor(img)
            
            # Extract results
            instances = outputs["instances"]
            masks = instances.pred_masks.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            
            # Get class names
            class_names = [COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}" 
                          for class_id in classes]
            
            logger.info(f"Detected {len(classes)} objects")
            return masks, classes, scores, class_names
            
        except Exception as e:
            logger.error(f"Error in image segmentation: {e}")
            return np.array([]), np.array([]), np.array([]), []
    
    def visualize_predictions(self, 
                            image_input: Union[str, np.ndarray, Image.Image],
                            show_boxes: bool = True,
                            show_masks: bool = True,
                            show_labels: bool = True) -> np.ndarray:
        """
        Visualize segmentation results on the image.
        
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
                img = cv2.imread(image_input)
                if img is None:
                    raise ValueError(f"Could not load image: {image_input}")
            elif isinstance(image_input, Image.Image):
                img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                img = image_input
            else:
                raise ValueError("image_input must be a file path, numpy array, or PIL Image")
            
            # Perform inference
            outputs = self.predictor(img)
            
            # Create visualizer
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
            
            # Create visualization
            instances = outputs["instances"]
            vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))
            
            # Convert back to BGR for OpenCV
            vis_image = vis_output.get_image()[:, :, ::-1]
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
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
                detections.append({
                    "object_id": i,
                    "class_name": class_name,
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "mask_area": int(np.sum(masks[i]))
                })
            
            summary = {
                "total_objects": len(classes),
                "object_counts": object_counts,
                "detections": detections,
                "average_confidence": float(np.mean(scores))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating detection summary: {e}")
            return {"total_objects": 0, "object_counts": {}, "detections": [], "error": str(e)}

# Initialize global segmenter instance
_segmenter = None

def get_segmenter() -> Optional[ImageSegmenter]:
    """
    Get or create a global ImageSegmenter instance.
    
    Returns:
        ImageSegmenter: Global segmenter instance, or None if Detectron2 is not available
    """
    global _segmenter
    if not DETECTRON2_AVAILABLE:
        return None
    
    if _segmenter is None:
        _segmenter = ImageSegmenter()
    return _segmenter

def segment_image(image_path: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to segment an image.
    
    Args:
        image_path (Union[str, np.ndarray, Image.Image]): Image path, numpy array, or PIL Image
        
    Returns:
        Tuple containing masks and classes
    """
    if not DETECTRON2_AVAILABLE:
        logger.warning("Detectron2 not available. Segmentation functionality is disabled.")
        return np.array([]), np.array([])
    
    segmenter = get_segmenter()
    if segmenter is None:
        logger.error("Could not initialize segmenter")
        return np.array([]), np.array([])
    
    masks, classes, _, _ = segmenter.segment_image(image_path)
    return masks, classes

def visualize_segmentation(image_path: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convenience function to visualize segmentation results.
    
    Args:
        image_path (Union[str, np.ndarray, Image.Image]): Image path, numpy array, or PIL Image
        
    Returns:
        np.ndarray: Image with visualizations
    """
    segmenter = get_segmenter()
    if segmenter is None:
        logger.error("Detectron2 not available for segmentation")
        if isinstance(image_path, str):
            return cv2.imread(image_path)
        return image_path
    
    return segmenter.visualize_predictions(image_path)

if __name__ == "__main__":
    # Test the segmentation functionality
    import sys
    
    if not DETECTRON2_AVAILABLE:
        print("Detectron2 is not available. Please install detectron2 to test segmentation.")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            masks, classes = segment_image(image_path)
            print(f"Detected {len(classes)} objects")
            
            # Get detailed summary
            segmenter = get_segmenter()
            summary = segmenter.get_detection_summary(image_path)
            print("Detection Summary:")
            print(f"Total Objects: {summary['total_objects']}")
            print(f"Object Counts: {summary['object_counts']}")
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python segmentation.py <image_path>")