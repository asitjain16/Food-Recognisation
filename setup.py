#!/usr/bin/env python3
"""
Setup script for Food Recognition App
This script downloads the dataset and sets up the environment
"""

import os
import sys
import subprocess
import kagglehub

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f" Error installing requirements: {e}")
        return False
    return True

def download_dataset():
    """Download the Food-101 dataset"""
    print("Downloading Food-101 dataset...")
    try:
        path = kagglehub.dataset_download("dansbecker/food-101")
        print(f" Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f" Error downloading dataset: {e}")
        return None

def create_directories():
    """Create necessary directories"""
    directories = ['static', 'uploads', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f" Created directory: {directory}")

def main():
    print(" Setting up Food Recognition App...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print(" Setup failed during requirements installation")
        return
    
    # Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        print("  Dataset download failed, but app can still run with fallback classes")
    
    print("\n" + "=" * 50)
    print(" Setup completed!")
    print("\nTo run the application:")
    print("  python app.py")
    print("\nThen open your browser to: http://localhost:5000")

if __name__ == "__main__":

    main()
