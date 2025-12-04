#!/usr/bin/env python
"""
Demo script for ImageCapSeg project.
This script demonstrates the core functionality without the Streamlit interface.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

def demo_captioning():
    """Demonstrate image captioning functionality."""
    print("\n" + "="*60)
    print("🖼️  IMAGE CAPTIONING DEMO")
    print("="*60)
    
    try:
        from captioning import generate_caption, generate_conditional_caption
        
        # Check for test images
        test_dir = Path("test_images")
        if not test_dir.exists():
            print("❌ test_images directory not found")
            print("Please add some test images to test_images/ directory")
            return
        
        # Find image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(test_dir.glob(f"*{ext}")))
            image_files.extend(list(test_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print("❌ No image files found in test_images/")
            print("Please add some test images (jpg, png, bmp) to test_images/ directory")
            return
        
        # Process first image
        image_path = str(image_files[0])
        print(f"📷 Processing: {image_path}")
        
        # Generate regular caption
        start_time = time.time()
        caption = generate_caption(image_path)
        caption_time = time.time() - start_time
        
        print(f"\n✨ Generated Caption ({caption_time:.2f}s):")
        print(f"   \"{caption}\"")
        
        # Generate conditional caption
        start_time = time.time()
        conditional_caption = generate_conditional_caption(image_path, "a photo of")
        conditional_time = time.time() - start_time
        
        print(f"\n✨ Conditional Caption ({conditional_time:.2f}s):")
        print(f"   \"{conditional_caption}\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in captioning demo: {e}")
        return False

def demo_segmentation():
    """Demonstrate image segmentation functionality."""
    print("\n" + "="*60)
    print("🎯 IMAGE SEGMENTATION DEMO")
    print("="*60)
    
    try:
        from segmentation import segment_image, get_segmenter
        
        # Check if Detectron2 is available
        segmenter = get_segmenter()
        if segmenter is None:
            print("❌ Detectron2 not available")
            print("Image segmentation requires detectron2 installation")
            return False
        
        # Check for test images
        test_dir = Path("test_images")
        if not test_dir.exists():
            print("❌ test_images directory not found")
            return False
        
        # Find image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(test_dir.glob(f"*{ext}")))
            image_files.extend(list(test_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print("❌ No image files found in test_images/")
            return False
        
        # Process first image
        image_path = str(image_files[0])
        print(f"📷 Processing: {image_path}")
        
        start_time = time.time()
        masks, classes = segment_image(image_path)
        segmentation_time = time.time() - start_time
        
        print(f"\n🎯 Segmentation Results ({segmentation_time:.2f}s):")
        print(f"   Detected {len(classes)} objects")
        
        if len(classes) > 0:
            # Get detailed summary
            summary = segmenter.get_detection_summary(image_path)
            print(f"\n📊 Object Summary:")
            for obj_type, count in summary.get('object_counts', {}).items():
                print(f"   - {obj_type.title()}: {count}")
            
            print(f"\n📋 Individual Detections:")
            for i, detection in enumerate(summary.get('detections', [])):
                print(f"   {i+1}. {detection['class_name'].title()} "
                      f"(confidence: {detection['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in segmentation demo: {e}")
        return False

def demo_system_info():
    """Display system information."""
    print("\n" + "="*60)
    print("💻 SYSTEM INFORMATION")
    print("="*60)
    
    try:
        from utils import ModelUtils
        
        device_info = ModelUtils.get_device_info()
        print(f"🖥️  Current Device: {device_info['current_device'].upper()}")
        print(f"🚀 CUDA Available: {device_info['cuda_available']}")
        
        if device_info['cuda_available']:
            print(f"🎮 GPU: {device_info.get('cuda_device_name', 'Unknown')}")
            print(f"🔧 CUDA Version: {device_info.get('cuda_version', 'Unknown')}")
            
            memory_info = device_info.get('cuda_memory', {})
            if memory_info:
                total_gb = memory_info.get('total', 0) / 1024**3
                allocated_gb = memory_info.get('allocated', 0) / 1024**3
                print(f"💾 GPU Memory: {allocated_gb:.1f}GB / {total_gb:.1f}GB")
        
        print(f"🐍 Python: {sys.version.split()[0]}")
        
        # Check package versions
        packages = ['torch', 'transformers', 'streamlit', 'opencv-python', 'pillow']
        print(f"\n📦 Package Versions:")
        
        for package in packages:
            try:
                if package == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                    name = 'OpenCV'
                elif package == 'torch':
                    import torch
                    version = torch.__version__
                    name = 'PyTorch'
                elif package == 'transformers':
                    import transformers
                    version = transformers.__version__
                    name = 'Transformers'
                elif package == 'streamlit':
                    import streamlit
                    version = streamlit.__version__
                    name = 'Streamlit'
                elif package == 'pillow':
                    import PIL
                    version = PIL.__version__
                    name = 'Pillow'
                
                print(f"   ✅ {name}: {version}")
                
            except ImportError:
                print(f"   ❌ {package}: Not installed")
        
        # Check Detectron2
        try:
            import detectron2
            print(f"   ✅ Detectron2: Available")
        except ImportError:
            print(f"   ❌ Detectron2: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting system info: {e}")
        return False

def main():
    """Run the complete demo."""
    print("🚀 ImageCapSeg Demo Script")
    print("This script demonstrates the core functionality of ImageCapSeg")
    
    # Display system info
    demo_system_info()
    
    # Run demos
    captioning_success = demo_captioning()
    segmentation_success = demo_segmentation()
    
    # Summary
    print("\n" + "="*60)
    print("📋 DEMO SUMMARY")
    print("="*60)
    
    print(f"📝 Image Captioning: {'✅ Working' if captioning_success else '❌ Failed'}")
    print(f"🎯 Image Segmentation: {'✅ Working' if segmentation_success else '❌ Failed'}")
    
    if captioning_success or segmentation_success:
        print(f"\n🎉 Demo completed successfully!")
        print(f"To run the full web application:")
        print(f"   streamlit run app.py")
    else:
        print(f"\n⚠️  Demo encountered issues.")
        print(f"Please check the installation and try running:")
        print(f"   python setup.py")

if __name__ == "__main__":
    main()