#!/usr/bin/env python3
"""
Quick test script for ImageCapSeg application
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        from src.captioning import generate_caption
        print("✅ Captioning module imported successfully")
    except ImportError as e:
        print(f"❌ Captioning module import failed: {e}")
        return False
    
    try:
        from src.yolo_segmentation import segment_image
        print("✅ YOLO segmentation module imported successfully")
    except ImportError as e:
        print(f"❌ YOLO segmentation module import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if required files and directories exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "app.py",
        "src/captioning.py",
        "src/yolo_segmentation.py",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    # Check if directories exist
    required_dirs = ["src"]
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ directory exists")
        else:
            print(f"❌ {dir_path}/ directory missing")
            return False
    
    # Optional directories (will be created automatically)
    optional_dirs = ["models", "uploads", "temp_images"]
    for dir_path in optional_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ directory exists")
        else:
            print(f"⚠️  {dir_path}/ directory missing (will be created automatically)")
    
    return True

def test_app_syntax():
    """Test if app.py has valid Python syntax"""
    print("\n🐍 Testing app.py syntax...")
    
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Compile to check for syntax errors
        compile(content, "app.py", "exec")
        print("✅ app.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ImageCapSeg Application Test")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: File structure
    if test_file_structure():
        tests_passed += 1
    
    # Test 2: App syntax
    if test_app_syntax():
        tests_passed += 1
    
    # Test 3: Imports (only if we're in the right environment)
    if test_imports():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your app is ready to run.")
        print("💡 To start the app, run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        print("💡 Make sure you're in the correct Python environment with all dependencies installed.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)