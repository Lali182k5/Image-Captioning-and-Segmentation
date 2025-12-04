#!/usr/bin/env python
"""
Setup script for ImageCapSeg project.
This script helps with the initial setup and dependency installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command with error handling."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False

def detect_gpu():
    """Detect if NVIDIA GPU is available."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
    except:
        pass
    
    print("ℹ️  No NVIDIA GPU detected (will use CPU)")
    return False

def install_pytorch(gpu_available=False):
    """Install PyTorch with appropriate configuration."""
    if gpu_available:
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        description = "Installing PyTorch with CUDA support"
    else:
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        description = "Installing PyTorch (CPU version)"
    
    return run_command(command, description)

def install_detectron2(gpu_available=False):
    """Install Detectron2."""
    if gpu_available:
        command = "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"
        description = "Installing Detectron2 with CUDA support"
    else:
        command = "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html"
        description = "Installing Detectron2 (CPU version)"
    
    success = run_command(command, description)
    
    if not success:
        print("\n⚠️  Detectron2 installation failed. Trying alternative method...")
        alt_command = "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
        success = run_command(alt_command, "Installing Detectron2 from source")
    
    return success

def install_requirements():
    """Install other requirements."""
    command = "pip install -r requirements.txt"
    description = "Installing other requirements"
    return run_command(command, description)

def create_directories():
    """Create necessary directories."""
    directories = [
        "models/caption_model",
        "models/segmentation_model",
        "test_images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_installation():
    """Test if all components are installed correctly."""
    print(f"\n{'='*50}")
    print("Testing installation...")
    print(f"{'='*50}")
    
    # Test imports
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found")
        return False
    
    try:
        import detectron2
        print("✅ Detectron2 available")
    except ImportError:
        print("⚠️  Detectron2 not found (segmentation will not work)")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  CUDA not available (using CPU)")
    
    print("\n🎉 Installation test completed!")
    return True

def main():
    """Main setup function."""
    print("🚀 ImageCapSeg Setup Script")
    print("This script will install all necessary dependencies for the ImageCapSeg project.")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect GPU
    gpu_available = detect_gpu()
    
    # Create directories
    print(f"\n{'='*50}")
    print("Creating project directories...")
    print(f"{'='*50}")
    create_directories()
    
    # Install PyTorch
    if not install_pytorch(gpu_available):
        print("❌ Failed to install PyTorch")
        sys.exit(1)
    
    # Install other requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Install Detectron2
    print("\n⚠️  Installing Detectron2 (this may take a while)...")
    detectron2_success = install_detectron2(gpu_available)
    
    if not detectron2_success:
        print("⚠️  Detectron2 installation failed. Image segmentation will not be available.")
        print("You can still use the image captioning features.")
    
    # Test installation
    test_installation()
    
    print(f"\n{'='*60}")
    print("🎉 Setup Complete!")
    print(f"{'='*60}")
    print("\nTo run the application:")
    print("streamlit run app.py")
    print("\nThen open your browser to: http://localhost:8501")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()