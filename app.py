"""
ImageCapSeg - AI-Powered Image Captioning and Segmentation
Modern web application with BLIP captioning and YOLOv8 segmentation.
"""

import streamlit as st
import sys
import os
from PIL import Image
import numpy as np
import cv2
from typing import Optional, List, Dict, Any
import time
import io
import base64
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
try:
    from src.captioning import generate_caption, generate_conditional_caption, get_captioner
    from src.utils import (
        ImageProcessor, ModelUtils, StreamlitUtils, FileManager,
        validate_image_file, get_supported_formats, check_gpu_availability
    )
    
    # Try to import segmentation modules - first try YOLO, then Detectron2
    try:
        from src.yolo_segmentation import segment_image, visualize_segmentation, get_segmenter
        SEGMENTATION_BACKEND = "YOLO"
    except ImportError:
        try:
            from src.segmentation import segment_image, visualize_segmentation, get_segmenter
            SEGMENTATION_BACKEND = "Detectron2"
        except ImportError:
            # Create dummy functions if no segmentation backend is available
            def segment_image(*args, **kwargs):
                return np.array([]), np.array([])
            def visualize_segmentation(*args, **kwargs):
                return None
            def get_segmenter():
                return None
            SEGMENTATION_BACKEND = "None"
    
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ImageCapSeg - AI Image Analysis",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with dark mode and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-in;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 0.5rem;
    }
    
    .hero-tech {
        font-size: 0.9rem;
        opacity: 0.8;
        background: rgba(255,255,255,0.1);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        margin-top: 1rem;
    }
    
    .feature-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .result-container {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    .tag {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        animation: fadeIn 0.5s ease-in;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Side-by-side layout */
    .side-by-side-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    @media (max-width: 768px) {
        .side-by-side-container {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
    }
    
    .side-by-side-left,
    .side-by-side-right {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .result-card {
        background: rgba(102, 126, 234, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .result-card h4 {
        color: #667eea;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .objects-list {
        margin-top: 1rem;
    }
    
    .object-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(102, 126, 234, 0.1);
        color: #495057;
    }
    
    .object-item:last-child {
        border-bottom: none;
    }
    
    /* Expandable tags */
    .expandable-tag {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .expandable-tag:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .tag-header {
        padding: 1rem 1.5rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.3s ease;
        user-select: none;
    }
    
    .tag-header:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    .tag-icon {
        font-size: 0.8rem;
        transition: transform 0.3s ease;
        width: 1rem;
        text-align: center;
    }
    
    .tag-icon.expanded {
        transform: rotate(90deg);
    }
    
    .tag-text {
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .tag-content {
        padding: 0 1.5rem 1rem 1.5rem;
        border-top: 1px solid rgba(102, 126, 234, 0.1);
        animation: slideDown 0.3s ease;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            max-height: 0;
        }
        to {
            opacity: 1;
            max-height: 200px;
        }
    }
    
    .download-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .download-btn:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        color: white;
        text-decoration: none;
    }
</style>

<script>
function toggleTag(tagId) {
    const content = document.getElementById('content-' + tagId);
    const icon = document.getElementById('icon-' + tagId);
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.classList.add('expanded');
        icon.textContent = '▼';
    } else {
        content.style.display = 'none';
        icon.classList.remove('expanded');
        icon.textContent = '▶';
    }
}
</script>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'caption_result' not in st.session_state:
        st.session_state.caption_result = None
    if 'segmentation_result' not in st.session_state:
        st.session_state.segmentation_result = None
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = {}
    if 'visualization_image' not in st.session_state:
        st.session_state.visualization_image = None

def create_download_link(img, filename, text):
    """Create a download link for images."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="text-decoration: none; background: linear-gradient(45deg, #28a745, #20c997); color: white; padding: 0.8rem 1.5rem; border-radius: 25px; font-weight: 600; display: inline-block; margin: 0.5rem;">{text}</a>'
    return href

def create_file_download_link(file_path, download_name):
    """Create a download link for files"""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:file/octet-stream;base64,{b64}" download="{download_name}" class="download-btn">📥 Download {download_name}</a>'

def create_side_by_side_layout(left_content, right_content):
    """Create a side-by-side layout for results"""
    return f"""
    <div class="side-by-side-container">
        <div class="side-by-side-left">
            {left_content}
        </div>
        <div class="side-by-side-right">
            {right_content}
        </div>
    </div>
    """

def create_expandable_tag(tag_text, content):
    """Create an expandable tag with content"""
    tag_id = tag_text.replace(" ", "-").lower().replace("📋", "").replace("🎯", "").replace("📈", "").strip()
    return f"""
    <div class="expandable-tag">
        <div class="tag-header" onclick="toggleTag('{tag_id}')">
            <span class="tag-icon" id="icon-{tag_id}">▶</span>
            <span class="tag-text">{tag_text}</span>
        </div>
        <div class="tag-content" id="content-{tag_id}" style="display: none;">
            {content}
        </div>
    </div>
    """

def display_header():
    """Display the modern hero section."""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🖼️ Image Captioning and Segmentation</div>
        <div class="hero-subtitle">AI-Powered Image Captioning & Segmentation</div>
        <div class="hero-tech">✨ Powered by BLIP & YOLOv8 ✨</div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with app information and settings."""
    with st.sidebar:
        st.header("🚀 App Information")
        
        # Device information
        with st.expander("💻 System Info"):
            device_info = ModelUtils.get_device_info()
            st.write(f"**Device:** {device_info['current_device'].upper()}")
            if device_info['cuda_available']:
                st.write(f"**GPU:** {device_info.get('cuda_device_name', 'Unknown')}")
                st.write(f"**CUDA Version:** {device_info.get('cuda_version', 'Unknown')}")
            
            # Check segmentation availability
            segmenter = get_segmenter()
            if segmenter is not None:
                st.write(f"**Segmentation:** ✅ Available ({SEGMENTATION_BACKEND})")
            else:
                st.write("**Segmentation:** ❌ Not Available")
            
            st.write("**Captioning:** ✅ Available (BLIP)")
            st.write(f"**Supported Formats:** {', '.join(get_supported_formats())}")
        
        # Processing options
        st.header("⚙️ Processing Options")
        
        # Caption settings
        st.subheader("📝 Caption Settings")
        max_caption_length = st.slider("Max Caption Length", 10, 50, 30, key="max_caption_length")
        use_conditional_caption = st.checkbox("Use Conditional Captioning", False, key="use_conditional_caption")
        if use_conditional_caption:
            caption_prompt = st.text_input("Caption Prompt", "a photo of", key="caption_prompt")
        else:
            caption_prompt = ""
        
        # Segmentation settings
        st.subheader("🎯 Segmentation Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, key="confidence_threshold")
        show_visualization = st.checkbox("Show Segmentation Overlay", True, key="show_visualization")
        
        # Image enhancement
        st.subheader("🎨 Image Enhancement")
        enhance_image = st.checkbox("Enable Image Enhancement", False, key="enhance_image")
        if enhance_image:
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, key="brightness")
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, key="contrast")
            saturation = st.slider("Saturation", 0.5, 2.0, 1.0, key="saturation")
        else:
            brightness = contrast = saturation = 1.0
        
        return {
            'max_caption_length': max_caption_length,
            'use_conditional_caption': use_conditional_caption,
            'caption_prompt': caption_prompt,
            'confidence_threshold': confidence_threshold,
            'show_visualization': show_visualization,
            'enhance_image': enhance_image,
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation
        }

def process_uploaded_image(uploaded_file, settings: Dict[str, Any]):
    """Process the uploaded image with captioning and segmentation."""
    if uploaded_file is None:
        return
    
    # Save uploaded file temporarily with unique name
    import time
    temp_path = f"temp_uploaded_image_{int(time.time())}.jpg"
    try:
        # Reset file pointer in case it was read before
        uploaded_file.seek(0)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Validate the image
        if not validate_image_file(temp_path):
            st.error("Invalid image file. Please upload a valid image.")
            return
        
        # Load and process the image
        original_image = Image.open(temp_path).convert("RGB")
        processed_image = original_image
        
        # Apply enhancements if enabled
        if settings['enhance_image']:
            processed_image = ImageProcessor.enhance_image(
                processed_image,
                brightness=settings['brightness'],
                contrast=settings['contrast'],
                saturation=settings['saturation']
            )
        
        st.session_state.processed_image = processed_image
        
        # Display the image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_image, width='stretch')
        
        if settings['enhance_image'] and processed_image != original_image:
            with col2:
                st.subheader("✨ Enhanced Image")
                st.image(processed_image, width='stretch')
        
        # Process captioning and segmentation
        process_image_analysis(temp_path, settings)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_image_analysis(image_path: str, settings: Dict[str, Any]):
    """Process image with both captioning and segmentation."""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    
    # Image Captioning
    status_text.text("Generating image caption...")
    progress_bar.progress(20)
    
    try:
        start_time = time.time()
        
        if settings['use_conditional_caption'] and settings['caption_prompt']:
            caption = generate_conditional_caption(
                image_path, 
                settings['caption_prompt'], 
                max_length=settings['max_caption_length']
            )
        else:
            caption = generate_caption(image_path, max_length=settings['max_caption_length'])
        
        caption_time = time.time() - start_time
        results['caption'] = caption
        results['caption_time'] = caption_time
        
    except Exception as e:
        results['caption'] = f"Error generating caption: {e}"
        results['caption_time'] = 0
    
    progress_bar.progress(60)
    status_text.text("Performing image segmentation...")
    
    # Image Segmentation
    try:
        start_time = time.time()
        segmenter = get_segmenter()
        
        if segmenter is not None:
            masks, classes, scores, class_names = segmenter.segment_image(image_path)
            segmentation_summary = segmenter.get_detection_summary(image_path)
            
            # Filter by confidence threshold
            valid_detections = [
                i for i, score in enumerate(scores) 
                if score >= settings['confidence_threshold']
            ]
            
            filtered_masks = masks[valid_detections] if len(valid_detections) > 0 else np.array([])
            filtered_classes = classes[valid_detections] if len(valid_detections) > 0 else np.array([])
            filtered_scores = scores[valid_detections] if len(valid_detections) > 0 else np.array([])
            filtered_class_names = [class_names[i] for i in valid_detections]
            
            segmentation_time = time.time() - start_time
            
            results['segmentation'] = {
                'masks': filtered_masks,
                'classes': filtered_classes,
                'scores': filtered_scores,
                'class_names': filtered_class_names,
                'summary': segmentation_summary,
                'total_objects': len(filtered_classes)
            }
            results['segmentation_time'] = segmentation_time
            
        else:
            if SEGMENTATION_BACKEND == "None":
                results['segmentation'] = {
                    'error': 'Segmentation not available. YOLO (YOLOv8) has been installed but may need initialization. Try refreshing the page or check the logs.'
                }
            else:
                results['segmentation'] = {
                    'error': f'{SEGMENTATION_BACKEND} segmentation backend failed to initialize. Check the logs for details.'
                }
            results['segmentation_time'] = 0
            
    except Exception as e:
        results['segmentation'] = {'error': f'Segmentation failed: {e}'}
        results['segmentation_time'] = 0
    
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    
    # Store results
    st.session_state.caption_result = results.get('caption', '')
    st.session_state.segmentation_result = results.get('segmentation', {})
    st.session_state.processing_times = {
        'caption': results.get('caption_time', 0),
        'segmentation': results.get('segmentation_time', 0)
    }
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    display_results(image_path, settings)

def display_results(image_path: str, settings: Dict[str, Any]):
    """Display the processing results."""
    
    st.header("📊 Analysis Results")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>⏱️ Caption Time</h3>
            <h2>{:.2f}s</h2>
        </div>
        """.format(st.session_state.processing_times.get('caption', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Segmentation Time</h3>
            <h2>{:.2f}s</h2>
        </div>
        """.format(st.session_state.processing_times.get('segmentation', 0)), unsafe_allow_html=True)
    
    with col3:
        seg_result = st.session_state.segmentation_result
        object_count = seg_result.get('total_objects', 0) if not seg_result.get('error') else 0
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Objects Found</h3>
            <h2>{}</h2>
        </div>
        """.format(object_count), unsafe_allow_html=True)
    
    # Caption Results
    st.subheader("📝 Image Caption")
    if st.session_state.caption_result:
        st.markdown(f"""
        <div class="result-container">
            <h4>Generated Caption:</h4>
            <p style="font-size: 1.2em; font-style: italic; color: #2c3e50;">
                "{st.session_state.caption_result}"
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Segmentation Results
    st.subheader("🎯 Object Detection & Segmentation")
    seg_result = st.session_state.segmentation_result
    
    if seg_result.get('error'):
        st.warning("🔧 **Segmentation Not Available**")
        st.info("""
        **Detectron2 is not installed.** Image captioning is working perfectly, but segmentation requires additional setup.
        
        **To enable segmentation (optional):**
        1. Install Visual Studio Build Tools for C++
        2. Run: `pip install git+https://github.com/facebookresearch/detectron2.git`
        
        **For now, you can enjoy full image captioning functionality!** 🎉
        """)
    elif seg_result.get('total_objects', 0) > 0:
        
        # Display segmentation visualization if enabled
        if settings['show_visualization']:
            try:
                vis_image = visualize_segmentation(image_path)
                if vis_image is not None:
                    st.subheader("🖼️ Segmentation Visualization")
                    # Convert BGR to RGB for Streamlit
                    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    st.image(vis_image_rgb, width='stretch')
            except Exception as e:
                st.warning(f"Could not generate visualization: {e}")
        
        # Display detection details
        st.markdown("""
        <div class="result-container">
            <h4>Detected Objects:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create detection summary
        detections_df_data = []
        for i, (class_name, score) in enumerate(zip(seg_result['class_names'], seg_result['scores'])):
            detections_df_data.append({
                'Object': class_name.title(),
                'Confidence': f"{score:.3f}",
                'ID': i + 1
            })
        
        if detections_df_data:
            st.table(detections_df_data)
        
        # Object count summary
        summary = seg_result.get('summary', {})
        if summary.get('object_counts'):
            st.subheader("📈 Object Count Summary")
            for obj_type, count in summary['object_counts'].items():
                st.write(f"**{obj_type.title()}**: {count}")
    
    else:
        st.info("No objects detected with the current confidence threshold.")

def display_sample_images():
    """Display sample images for testing."""
    st.header("🖼️ Try Sample Images")
    
    # Check if test images directory exists
    test_images_dir = "test_images"
    if os.path.exists(test_images_dir):
        sample_images = FileManager.get_image_files(test_images_dir)
        
        if sample_images:
            selected_sample = st.selectbox(
                "Choose a sample image:",
                ["None"] + [os.path.basename(img) for img in sample_images],
                key="selected_sample"
            )
            
            if selected_sample != "None":
                sample_path = os.path.join(test_images_dir, selected_sample)
                st.image(sample_path, caption=f"Sample: {selected_sample}", width='stretch')
                
                if st.button("Process Sample Image", key="process_sample"):
                    # Get settings from session state or use defaults
                    settings = {
                        'max_caption_length': st.session_state.get('max_caption_length', 30),
                        'use_conditional_caption': st.session_state.get('use_conditional_caption', False),
                        'caption_prompt': st.session_state.get('caption_prompt', ''),
                        'confidence_threshold': st.session_state.get('confidence_threshold', 0.5),
                        'show_visualization': st.session_state.get('show_visualization', True),
                        'enhance_image': st.session_state.get('enhance_image', False),
                        'brightness': st.session_state.get('brightness', 1.0),
                        'contrast': st.session_state.get('contrast', 1.0),
                        'saturation': st.session_state.get('saturation', 1.0)
                    }
                    process_image_analysis(sample_path, settings)
        else:
            st.info("No sample images found in test_images directory.")
    else:
        st.info("test_images directory not found. Upload your own images to test the application.")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar and get settings
    settings = display_sidebar()
    
    # Main content
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=list(get_supported_formats()),
        help="Upload an image for AI analysis (captioning and segmentation)",
        key="main_file_uploader"
    )
    
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file, settings)
    else:
        # Display sample images section
        display_sample_images()
        
        # Display feature information
        st.header("✨ Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card" style="background: rgba(255, 255, 255, 0.95); color: #2c3e50;">
                <h3 style="color: #667eea; margin-bottom: 1rem;">📝 Image Captioning</h3>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid #eee;">✅ Powered by BLIP model</li>
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid #eee;">✅ Natural language descriptions</li>
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid #eee;">✅ Conditional captioning support</li>
                    <li style="padding: 0.5rem 0;">✅ High accuracy results</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card" style="background: rgba(255, 255, 255, 0.95); color: #2c3e50;">
                <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Object Segmentation</h3>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid #eee;">✅ Powered by YOLOv8 model</li>
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid #eee;">✅ 80+ object categories (COCO)</li>
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid #eee;">✅ Instance segmentation masks</li>
                    <li style="padding: 0.5rem 0;">✅ Confidence scoring</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Usage instructions
        st.header("📋 How to Use")        
        st.markdown("""
        1. **Upload an image** using the file uploader above
        2. **Adjust settings** in the sidebar (optional)
        3. **Wait for processing** - both captioning and segmentation will run automatically
        4. **View results** including captions, detected objects, and visualizations
        5. **Try sample images** if available in the test_images directory
        """)

if __name__ == "__main__":
    main()