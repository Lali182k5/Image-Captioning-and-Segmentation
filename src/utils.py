"""
Utility Functions for Image Processing and Model Management
This module provides helper functions for image processing, validation, and common operations.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import List, Tuple, Union, Optional, Dict, Any
import base64
from io import BytesIO
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

class ImageProcessor:
    """
    A utility class for image processing operations.
    """
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validate if the given path points to a valid image file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            bool: True if valid image, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                return False
            
            # Check file extension
            _, ext = os.path.splitext(image_path.lower())
            if ext not in SUPPORTED_FORMATS:
                return False
            
            # Try to open the image
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            logger.warning(f"Image validation failed for {image_path}: {e}")
            return False
    
    @staticmethod
    def resize_image(image: Union[str, Image.Image, np.ndarray], 
                    target_size: Tuple[int, int], 
                    maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Resize an image to the target size.
        
        Args:
            image (Union[str, Image.Image, np.ndarray]): Input image
            target_size (Tuple[int, int]): Target width and height
            maintain_aspect_ratio (bool): Whether to maintain aspect ratio
            
        Returns:
            Image.Image: Resized PIL Image
        """
        try:
            # Convert to PIL Image
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image.copy()
            else:
                raise ValueError("Unsupported image type")
            
            if maintain_aspect_ratio:
                pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
            else:
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            # Return a blank image if resize fails
            return Image.new('RGB', target_size, color=(255, 255, 255))
    
    @staticmethod
    def enhance_image(image: Image.Image, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> Image.Image:
        """
        Enhance image with brightness, contrast, saturation, and sharpness adjustments.
        
        Args:
            image (Image.Image): Input PIL Image
            brightness (float): Brightness factor (1.0 = no change)
            contrast (float): Contrast factor (1.0 = no change)
            saturation (float): Saturation factor (1.0 = no change)
            sharpness (float): Sharpness factor (1.0 = no change)
            
        Returns:
            Image.Image: Enhanced PIL Image
        """
        try:
            enhanced_image = image.copy()
            
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(enhanced_image)
                enhanced_image = enhancer.enhance(saturation)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(sharpness)
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    @staticmethod
    def apply_filters(image: Image.Image, filter_type: str = "none") -> Image.Image:
        """
        Apply filters to the image.
        
        Args:
            image (Image.Image): Input PIL Image
            filter_type (str): Type of filter to apply
            
        Returns:
            Image.Image: Filtered PIL Image
        """
        try:
            if filter_type == "blur":
                return image.filter(ImageFilter.BLUR)
            elif filter_type == "sharpen":
                return image.filter(ImageFilter.SHARPEN)
            elif filter_type == "edge_enhance":
                return image.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_type == "smooth":
                return image.filter(ImageFilter.SMOOTH)
            elif filter_type == "emboss":
                return image.filter(ImageFilter.EMBOSS)
            else:
                return image
                
        except Exception as e:
            logger.error(f"Error applying filter {filter_type}: {e}")
            return image
    
    @staticmethod
    def convert_format(image: Image.Image, format: str = "RGB") -> Image.Image:
        """
        Convert image to specified format.
        
        Args:
            image (Image.Image): Input PIL Image
            format (str): Target format (RGB, RGBA, L, etc.)
            
        Returns:
            Image.Image: Converted PIL Image
        """
        try:
            return image.convert(format)
        except Exception as e:
            logger.error(f"Error converting image format: {e}")
            return image

class FileManager:
    """
    A utility class for file management operations.
    """
    
    @staticmethod
    def create_directory(directory_path: str) -> bool:
        """
        Create a directory if it doesn't exist.
        
        Args:
            directory_path (str): Path to the directory
            
        Returns:
            bool: True if directory was created or already exists
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def get_image_files(directory: str) -> List[str]:
        """
        Get list of image files in a directory.
        
        Args:
            directory (str): Directory path
            
        Returns:
            List[str]: List of image file paths
        """
        try:
            if not os.path.exists(directory):
                return []
            
            image_files = []
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file.lower())
                    if ext in SUPPORTED_FORMATS:
                        image_files.append(file_path)
            
            return sorted(image_files)
            
        except Exception as e:
            logger.error(f"Error getting image files from {directory}: {e}")
            return []
    
    @staticmethod
    def save_image(image: Image.Image, file_path: str, quality: int = 95) -> bool:
        """
        Save PIL Image to file.
        
        Args:
            image (Image.Image): PIL Image to save
            file_path (str): Output file path
            quality (int): JPEG quality (1-100)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Save image
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                image.save(file_path, 'JPEG', quality=quality)
            else:
                image.save(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to {file_path}: {e}")
            return False

class ModelUtils:
    """
    Utility functions for model management and inference.
    """
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get information about available compute devices.
        
        Returns:
            Dict[str, Any]: Device information
        """
        device_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        if torch.cuda.is_available():
            device_info["cuda_version"] = torch.version.cuda
            device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            device_info["cuda_memory"] = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "reserved": torch.cuda.memory_reserved(0)
            }
        
        return device_info
    
    @staticmethod
    def clear_memory():
        """
        Clear GPU memory cache.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    @staticmethod
    def benchmark_inference_speed(model_function, input_data, num_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark inference speed of a model function.
        
        Args:
            model_function: Function to benchmark
            input_data: Input data for the function
            num_iterations (int): Number of iterations to run
            
        Returns:
            Dict[str, float]: Timing statistics
        """
        import time
        
        times = []
        
        # Warm-up run
        try:
            _ = model_function(input_data)
        except Exception as e:
            logger.error(f"Warm-up run failed: {e}")
            return {"error": str(e)}
        
        # Benchmark runs
        for _ in range(num_iterations):
            start_time = time.time()
            try:
                _ = model_function(input_data)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Benchmark iteration failed: {e}")
                continue
        
        if not times:
            return {"error": "All benchmark iterations failed"}
        
        times = np.array(times)
        return {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "iterations": len(times)
        }

class StreamlitUtils:
    """
    Utility functions specifically for Streamlit applications.
    """
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string for Streamlit display.
        
        Args:
            image (Image.Image): PIL Image
            format (str): Image format for encoding
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            buffered = BytesIO()
            image.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""
    
    @staticmethod
    def create_download_link(data: bytes, filename: str, text: str = "Download") -> str:
        """
        Create a download link for Streamlit.
        
        Args:
            data (bytes): Data to download
            filename (str): Filename for download
            text (str): Link text
            
        Returns:
            str: HTML download link
        """
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
        return href
    
    @staticmethod
    def format_detection_results(detections: List[Dict]) -> str:
        """
        Format detection results for Streamlit display.
        
        Args:
            detections (List[Dict]): List of detection results
            
        Returns:
            str: Formatted results string
        """
        if not detections:
            return "No objects detected."
        
        result_lines = []
        for i, detection in enumerate(detections, 1):
            class_name = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0.0)
            result_lines.append(f"{i}. **{class_name}** (Confidence: {confidence:.2f})")
        
        return "\n".join(result_lines)

# Convenience functions
def validate_image_file(file_path: str) -> bool:
    """Convenience function to validate an image file."""
    return ImageProcessor.validate_image(file_path)

def resize_image_file(file_path: str, target_size: Tuple[int, int]) -> Image.Image:
    """Convenience function to resize an image file."""
    return ImageProcessor.resize_image(file_path, target_size)

def get_supported_formats() -> List[str]:
    """Get list of supported image formats."""
    return list(SUPPORTED_FORMATS)

def check_gpu_availability() -> bool:
    """Check if GPU is available for computation."""
    return torch.cuda.is_available()

if __name__ == "__main__":
    # Test utility functions
    print("=== ImageCapSeg Utilities Test ===")
    
    # Test device info
    device_info = ModelUtils.get_device_info()
    print(f"Device Info: {device_info}")
    
    # Test supported formats
    formats = get_supported_formats()
    print(f"Supported Formats: {formats}")
    
    # Test GPU availability
    gpu_available = check_gpu_availability()
    print(f"GPU Available: {gpu_available}")
    
    print("=== Test Complete ===")