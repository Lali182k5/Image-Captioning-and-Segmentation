# Sample Test Images

This directory contains sample images for testing the ImageCapSeg application.

## Adding Your Own Test Images

1. Place image files in this directory
2. Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP
3. Recommended resolution: 512x512 to 1024x1024 for optimal performance
4. The images will appear in the "Try Sample Images" section of the app

## Suggested Test Images

For the best demonstration of the application capabilities, consider adding images that contain:

### For Captioning Testing
- Scenic landscapes
- People in various activities
- Food and dining scenes
- Urban environments
- Animals in natural settings

### For Segmentation Testing
- Multiple people in a scene
- Various vehicles (cars, buses, motorcycles)
- Animals (cats, dogs, birds, etc.)
- Household objects and furniture
- Food items and kitchen scenes
- Sports activities and equipment

## Example Usage

Once you've added images to this directory:

1. Run the Streamlit app: `streamlit run app.py`
2. Scroll down to the "Try Sample Images" section
3. Select an image from the dropdown
4. Click "Process Sample Image" to analyze it

## Performance Tips

- Images larger than 2MB may take longer to process
- GPU acceleration significantly improves processing speed
- Consider resizing very large images (>4000px) before processing