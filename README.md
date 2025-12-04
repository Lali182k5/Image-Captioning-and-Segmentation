# 🖼️ ImageCapSeg - AI-Powered Image Analysis

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.25+-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive web application that combines **Image Captioning** and **Object Segmentation** using state-of-the-art AI models. Generate natural language descriptions and detect objects in images with pixel-perfect segmentation masks.

## 🎯 Features

### 📝 Image Captioning (BLIP)
- **Natural Language Generation**: Create human-readable descriptions of images
- **Conditional Captioning**: Generate captions based on specific prompts
- **Adjustable Length**: Control caption length (10-50 words)
- **High Accuracy**: Powered by Salesforce's BLIP model

### 🎯 Object Segmentation (Detectron2)
- **Instance Segmentation**: Detect and segment individual objects
- **80+ Object Classes**: COCO dataset categories (person, car, dog, etc.)
- **Confidence Scoring**: Filter detections by confidence threshold
- **Visual Overlays**: See segmentation masks overlaid on images

### 🚀 Web Interface
- **Streamlit UI**: Clean, intuitive web interface
- **Real-time Processing**: Upload and analyze images instantly
- **Interactive Controls**: Adjust settings and parameters
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Project Structure

```
ImageCapSeg/
│── models/
│   ├── caption_model/       # Optional fine-tuned captioning model
│   └── segmentation_model/  # Optional saved weights
│
│── src/
│   ├── captioning.py        # Image Captioning with BLIP
│   ├── segmentation.py      # Image Segmentation with Detectron2
│   └── utils.py             # Helper functions and utilities
│
│── app.py                   # Streamlit web application
│── test_images/             # Sample test images
│── requirements.txt         # Python dependencies
│── README.md               # This file
└── LICENSE                 # MIT License
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ImageCapSeg.git
cd ImageCapSeg
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n imagecapseg python=3.9
conda activate imagecapseg

# Or using venv
python -m venv imagecapseg
# Windows
imagecapseg\Scripts\activate
# Linux/Mac
source imagecapseg/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch (choose appropriate version for your system)
# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU version (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 4. Install Detectron2

Detectron2 requires special installation:

```bash
# For CPU
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# For GPU (CUDA 11.8)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Or build from source
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Usage

1. **Upload Image**: Click "Browse files" and select an image (JPG, PNG, etc.)
2. **Wait for Processing**: The app will automatically generate captions and detect objects
3. **View Results**: See the generated caption and detected objects with confidence scores
4. **Explore Visualizations**: View segmentation masks overlaid on your image

### Advanced Settings

Access the sidebar to customize processing:

#### 📝 Caption Settings
- **Max Caption Length**: Control output length (10-50 words)
- **Conditional Captioning**: Add prompts like "a photo of" or "this image shows"

#### 🎯 Segmentation Settings
- **Confidence Threshold**: Filter detections (0.1-0.9)
- **Show Visualization**: Toggle segmentation overlay display

#### 🎨 Image Enhancement
- **Brightness**: Adjust image brightness (0.5-2.0)
- **Contrast**: Enhance image contrast (0.5-2.0)
- **Saturation**: Modify color saturation (0.5-2.0)

### Sample Images

Place test images in the `test_images/` directory to quickly test the application with pre-loaded samples.

## 🛠️ Technical Details

### Models Used

#### BLIP (Image Captioning)
- **Model**: `Salesforce/blip-image-captioning-base`
- **Architecture**: Vision-Language Transformer
- **Training**: Large-scale web data
- **Capabilities**: Unconditional and conditional captioning

#### Detectron2 (Object Segmentation)
- **Model**: `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x`
- **Architecture**: Mask R-CNN with ResNet-50 backbone
- **Training**: COCO dataset (80 object classes)
- **Output**: Bounding boxes, masks, and class predictions

### Performance

| Operation | CPU Time | GPU Time | Memory Usage |
|-----------|----------|----------|--------------|
| Caption Generation | ~3-5s | ~1-2s | ~2GB |
| Object Segmentation | ~5-10s | ~2-3s | ~3GB |
| Image Preprocessing | <1s | <1s | ~500MB |

### System Requirements

#### Minimum
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 5GB free space
- **OS**: Windows 10, macOS 10.15, or Linux

#### Recommended
- **Python**: 3.9+
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 10GB+ free space

## 🎮 API Usage

You can also use the models programmatically:

```python
from src.captioning import generate_caption
from src.segmentation import segment_image

# Generate caption
caption = generate_caption("path/to/image.jpg")
print(f"Caption: {caption}")

# Segment image
masks, classes = segment_image("path/to/image.jpg")
print(f"Found {len(classes)} objects")
```

### Advanced API

```python
from src.captioning import ImageCaptioner
from src.segmentation import ImageSegmenter

# Initialize models
captioner = ImageCaptioner()
segmenter = ImageSegmenter(score_threshold=0.7)

# Generate conditional caption
caption = captioner.generate_conditional_caption(
    "image.jpg", 
    "a photo of", 
    max_length=25
)

# Get detailed segmentation results
masks, classes, scores, names = segmenter.segment_image("image.jpg")
summary = segmenter.get_detection_summary("image.jpg")
```

## 📊 Supported Object Classes

The segmentation model can detect 80 different object classes from the COCO dataset:

**People & Animals**: person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, bird

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Furniture**: chair, couch, bed, dining table, toilet

**Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, refrigerator

**Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, bottle, wine glass, cup, fork, knife, spoon, bowl

**Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Household**: backpack, umbrella, handbag, tie, suitcase, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

And many more! See the full list in `src/segmentation.py`.

## 🔧 Configuration

### Model Configuration

You can customize models by modifying the initialization parameters:

```python
# Custom captioning model
from src.captioning import ImageCaptioner
captioner = ImageCaptioner(
    model_name="Salesforce/blip-image-captioning-large"  # Use larger model
)

# Custom segmentation model
from src.segmentation import ImageSegmenter
segmenter = ImageSegmenter(
    model_config="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
    score_threshold=0.3
)
```

### Environment Variables

Set these environment variables for additional control:

```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export TORCH_HOME=/path/to/models  # Model cache directory
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Detectron2 Installation Failed
```bash
# Try installing build tools first
pip install pybind11
# Then install detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

#### 2. CUDA Out of Memory
- Reduce batch size or image resolution
- Close other GPU-intensive applications
- Use CPU mode: Set `device="cpu"` in model initialization

#### 3. Slow Performance
- Enable GPU acceleration if available
- Reduce image resolution before processing
- Lower confidence threshold for segmentation

#### 4. Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --force-reinstall
```

### Performance Optimization

1. **GPU Acceleration**: Install CUDA-compatible PyTorch
2. **Model Caching**: Models are cached after first download
3. **Image Preprocessing**: Resize large images before processing
4. **Memory Management**: Clear GPU cache between processing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Salesforce** for the BLIP image captioning model
- **Facebook Research** for Detectron2 and object segmentation
- **Streamlit** for the excellent web framework
- **PyTorch** and **Hugging Face** for the underlying ML infrastructure

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ImageCapSeg/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ImageCapSeg/discussions)
- **Email**: your.email@example.com

## 🚀 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

### Code Style

We use `black` for code formatting and `flake8` for linting:

```bash
pip install black flake8
black .
flake8 .
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Built with ❤️ using PyTorch, Transformers, and Streamlit

</div>