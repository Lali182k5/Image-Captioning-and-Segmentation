"""
Image Captioning Module using BLIP (Bootstrapping Language-Image Pre-training)
This module provides functionality to generate captions for images using the BLIP model.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import logging
from typing import Optional, Union
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCaptioner:
    """
    A class for generating captions for images using the BLIP model.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the ImageCaptioner with the specified model.
        
        Args:
            model_name (str): The name of the BLIP model to use
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the BLIP processor and model."""
        try:
            logger.info(f"Loading BLIP model: {self.model_name}")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_caption(self, 
                        image_input: Union[str, Image.Image], 
                        max_length: int = 30,
                        min_length: int = 5,
                        num_beams: int = 5) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image_input (Union[str, Image.Image]): Path to image file or PIL Image object
            max_length (int): Maximum length of generated caption
            min_length (int): Minimum length of generated caption
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated caption for the image
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ValueError("image_input must be a file path (str) or PIL Image object")
            
            # Process the image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the output
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"Error: Unable to generate caption - {str(e)}"
    
    def generate_conditional_caption(self, 
                                   image_input: Union[str, Image.Image],
                                   text_prompt: str,
                                   max_length: int = 30) -> str:
        """
        Generate a conditional caption based on a text prompt.
        
        Args:
            image_input (Union[str, Image.Image]): Path to image file or PIL Image object
            text_prompt (str): Text prompt to condition the caption generation
            max_length (int): Maximum length of generated caption
            
        Returns:
            str: Generated conditional caption
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ValueError("image_input must be a file path (str) or PIL Image object")
            
            # Process the image with text prompt
            inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.device)
            
            # Generate conditional caption
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=False
                )
            
            # Decode the output
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated conditional caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating conditional caption: {e}")
            return f"Error: Unable to generate conditional caption - {str(e)}"

# Initialize global captioner instance
_captioner = None

def get_captioner() -> ImageCaptioner:
    """
    Get or create a global ImageCaptioner instance.
    
    Returns:
        ImageCaptioner: Global captioner instance
    """
    global _captioner
    if _captioner is None:
        _captioner = ImageCaptioner()
    return _captioner

def generate_caption(image_path: Union[str, Image.Image], 
                    max_length: int = 30) -> str:
    """
    Convenience function to generate a caption for an image.
    
    Args:
        image_path (Union[str, Image.Image]): Path to image file or PIL Image object
        max_length (int): Maximum length of generated caption
        
    Returns:
        str: Generated caption for the image
    """
    captioner = get_captioner()
    return captioner.generate_caption(image_path, max_length=max_length)

def generate_conditional_caption(image_path: Union[str, Image.Image],
                               text_prompt: str,
                               max_length: int = 30) -> str:
    """
    Convenience function to generate a conditional caption for an image.
    
    Args:
        image_path (Union[str, Image.Image]): Path to image file or PIL Image object
        text_prompt (str): Text prompt to condition the caption generation
        max_length (int): Maximum length of generated caption
        
    Returns:
        str: Generated conditional caption for the image
    """
    captioner = get_captioner()
    return captioner.generate_conditional_caption(image_path, text_prompt, max_length=max_length)

if __name__ == "__main__":
    # Test the captioning functionality
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            caption = generate_caption(image_path)
            print(f"Caption: {caption}")
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python captioning.py <image_path>")