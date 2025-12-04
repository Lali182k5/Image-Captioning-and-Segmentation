"""
ImageCapSeg - AI-Powered Image Analysis Package

This package provides image captioning and segmentation functionality
using state-of-the-art models BLIP and Detectron2.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main functions for easy access
try:
    from .captioning import generate_caption, generate_conditional_caption, ImageCaptioner
    from .segmentation import segment_image, visualize_segmentation, ImageSegmenter
    from .utils import (
        ImageProcessor, ModelUtils, StreamlitUtils, FileManager,
        validate_image_file, get_supported_formats, check_gpu_availability
    )
    
    __all__ = [
        'generate_caption',
        'generate_conditional_caption', 
        'ImageCaptioner',
        'segment_image',
        'visualize_segmentation',
        'ImageSegmenter',
        'ImageProcessor',
        'ModelUtils',
        'StreamlitUtils',
        'FileManager',
        'validate_image_file',
        'get_supported_formats',
        'check_gpu_availability'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    __all__ = []