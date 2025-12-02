"""
ID Card Detection Training Script - Main Training Orchestrator
==============================================================  A training launcher (only starts training)

DESCRIPTION:
This script serves as the main entry point for training the ID card detection system.
It orchestrates the entire training pipeline by utilizing the YOLOIDDetector class
and provides a simplified interface for model training with predefined configurations.

KEY RESPONSIBILITIES:
- Import and initialize the YOLOIDDetector class
- Define dataset paths and training parameters
- Coordinate the dataset preparation process
- Execute model training with optimized settings
- Provide training progress monitoring and debugging
- Generate comprehensive training visualizations

WORKFLOW:
1. Import YOLOIDDetector class using direct file import
2. Set up dataset paths (images, labels, output directories)
3. Prepare dataset in YOLO format using detector.prepare_dataset()
4. Train the object detection model using detector.train_model()
5. Output training results and model information
6. Generate training metrics and visualizations

CONFIGURATION:
- Pre-defined paths for images, labels, and output directories
- Fixed training parameters (epochs=100, image_size=640)
- Automatic train/validation split (80/20)
- Enhanced data augmentation for better generalization

USAGE:
    Run directly from command line:
        python training_classifier.py
    
    The script will automatically:
    - Load the YOLOIDDetector class
    - Prepare the dataset
    - Train the model for 100 epochs with data augmentation
    - Output training progress and results
    - Generate comprehensive training plots

NOTE:
This script is designed for the specific ID card detection project structure
and expects certain directory paths to exist. Modify the base_path variable
according to your project structure.

DEPENDENCIES:
- Requires YOLOIDDetector.py in the specified path
- Same dependencies as YOLOIDDetector
- Additional: matplotlib, pandas for visualization

training_classifier.py (Driver Script)
         ↓
YOLOIDDetector.py (Core Engine)
         ↓
Ultralytics YOLO (Underlying Framework)
         ↓
Trained Model Output + Visualizations
"""

import os
import sys
import importlib.util

yoloid_file_path = '/Users/emmafarigoule/Desktop/CICR/OCR&YOLO/NewAttempt/ModelTraining/YOLOIDDetector.py'

print(f"🔍 Loading YOLOIDDetector from: {yoloid_file_path}")

# Check if the file exists
if not os.path.exists(yoloid_file_path):
    print(f"❌ YOLOIDDetector.py not found at: {yoloid_file_path}")
    sys.exit(1)

# Import the module directly from file
try:
    spec = importlib.util.spec_from_file_location("YOLOIDDetector", yoloid_file_path)
    yoloid_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(yoloid_module)
    YOLOIDDetector = yoloid_module.YOLOIDDetector
    print("✅ Successfully loaded YOLOIDDetector via direct file import")
    
except Exception as e:
    print(f"❌ Direct file import failed: {e}")
    sys.exit(1)

def train_id_detector():
    print("🚀 Starting ID Card and Picture Detector Training...")
    print("📊 Comprehensive metrics and visualizations will be generated")
    
    # Initialize your detector
    detector = YOLOIDDetector()
    
    # Define your paths - Updated to use your specific paths
    base_path = '/Users/emmafarigoule/Desktop/CICR/OCR&YOLO/id_card_detector'
    images_dir = '/Users/emmafarigoule/Desktop/CICR/OCR&YOLO/id_card_detector/my_images'
    labels_dir = '/Users/emmafarigoule/Desktop/CICR/OCR&YOLO/id_card_detector/my_labels'
    output_dir = '/Users/emmafarigoule/Desktop/CICR/OCR&YOLO/id_card_detector/id_card_dataset_v3'
    
    print(f"📁 Using paths:")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    print(f"   Output: {output_dir}")
    
    # Verify paths exist
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # Prepare dataset
    dataset_path = detector.prepare_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir
    )
    
    print(f"📁 Dataset prepared at: {dataset_path}")
    
    # Train the model with enhanced data augmentation
    print("🎯 Training YOLO model with data augmentation...")
    results = detector.train_model(
        dataset_dir=output_dir,
        epochs=50,
        imgsz=640,
        batch=16,                    # Batch size
        lr0=0.0005,                   # Reduced learning rate for stability
        augment=True,                # Enable data augmentation
        
        # Enhanced augmentation parameters
        hsv_h=0.015,                # Hue augmentation
        hsv_s=0.7,                  # Saturation augmentation  
        hsv_v=0.4,                  # Value augmentation
        degrees=2.0,               # Rotation degrees
        translate=0.1,              # Translation
        scale=0.1,                  # Scale augmentation
        fliplr=0.5,                 # Horizontal flip probability
        
        # Regularization to reduce overfitting
        dropout=0.2,                # Add dropout regularization
        weight_decay=0.001,        # Increased weight decay

        # Optimization
        patience=20,                # Early stopping patience
        verbose=True,
        save=True
    )
    
    model = detector.model
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("✅ Training completed!")
    print("📝 Classes: ID (class 0), Picture (class 1)")
    print("📈 Training metrics and plots have been generated automatically")

if __name__ == "__main__":
    train_id_detector()