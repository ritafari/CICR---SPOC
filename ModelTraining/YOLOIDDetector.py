"""
YOLOIDDetector - Object Detection Model for ID Cards
==================================================== Complete AI system (trains + detects)

DESCRIPTION:
This class provides a complete YOLO-based object detection pipeline specifically 
designed for detecting ID cards and Pictures in images and documents. It handles model 
initialization, dataset preparation in YOLO format, and model training.

KEY FEATURES:
- Load pre-trained YOLO models or custom trained models
- Automatically convert datasets to YOLO format (images + labels)
- Split data into train/validation sets (80/20)
- Generate YOLO dataset configuration YAML files
- Train object detection models with optimized defaults
- Comprehensive training metrics and visualization

ARCHITECTURE:
- Uses Ultralytics YOLOv8 as the underlying detection engine
- Supports both classification and detection tasks
- Handles image preprocessing and augmentation automatically

USAGE:
    detector = YOLOIDDetector()  # Load pre-trained model
    detector = YOLOIDDetector('path/to/custom_model.pt')  # Load custom model
    
    # Prepare dataset
    dataset_path = detector.prepare_dataset(
        images_dir='path/to/images',
        labels_dir='path/to/labels', 
        output_dir='dataset_output'
    )
    
    # Train model
    results = detector.train_model(
        dataset_dir=dataset_path,
        epochs=100,
        imgsz=640
    )

INPUT/OUTPUT:
- Input: Raw images and YOLO-format label files
- Output: Trained YOLO model weights and training metrics

DEPENDENCIES:
- ultralytics, opencv-python, numpy, PyMuPDF, matplotlib, pandas
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
import fitz  # PyMuPDF
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class YOLOIDDetector:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded trained model: {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')
            print("✅ Loaded pre-trained YOLOv8 model")
        
        # Store training results for analysis
        self.training_results = None
    
    def prepare_dataset(self, images_dir, labels_dir, output_dir='id_dataset'):
        """Prepare YOLO format dataset with two classes"""
        # Create dataset structure
        os.makedirs(f'{output_dir}/images/train', exist_ok=True)
        os.makedirs(f'{output_dir}/images/val', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/val', exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train/val (80/20)
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy images and labels
        for img_file in train_files:
            self._copy_to_yolo_format(img_file, images_dir, labels_dir, 
                                    f'{output_dir}/images/train', 
                                    f'{output_dir}/labels/train')
        
        for img_file in val_files:
            self._copy_to_yolo_format(img_file, images_dir, labels_dir, 
                                    f'{output_dir}/images/val', 
                                    f'{output_dir}/labels/val')
        
        # Create dataset.yaml with two classes
        self._create_dataset_yaml(output_dir)
        
        print(f"✅ Dataset prepared at: {output_dir}")
        print(f"📊 Training samples: {len(train_files)}")
        print(f"📊 Validation samples: {len(val_files)}")
        return output_dir
    
    def _copy_to_yolo_format(self, img_file, src_img_dir, src_label_dir, dest_img_dir, dest_label_dir):
        """Copy and convert to YOLO format"""
        # Copy image
        shutil.copy2(os.path.join(src_img_dir, img_file), 
                    os.path.join(dest_img_dir, img_file))
        
        # Copy corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label_path = os.path.join(src_label_dir, label_file)
        
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, os.path.join(dest_label_dir, label_file))
    
    def _create_dataset_yaml(self, dataset_dir):
        """Create YOLO dataset configuration file with two classes"""
        yaml_content = f"""
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

nc: 2  # number of classes
names: ['ID', 'Picture']  # class names - ID (class 0), Picture (class 1)
"""
        with open(f'{dataset_dir}/dataset.yaml', 'w') as f:
            f.write(yaml_content)
    
    def train_model(self, dataset_dir, epochs=100, imgsz=640, **kwargs):
        """Train YOLO model with flexible parameters and data augmentation"""
        print("🚀 Starting YOLO model training...")
        
        # If dataset_dir is a directory, use the YAML file inside it
        if os.path.isdir(dataset_dir):
            data_path = f'{dataset_dir}/dataset.yaml'
        else:
            data_path = dataset_dir
        
        print(f"📄 Using dataset: {data_path}")
        
        # Enhanced training arguments with data augmentation
        train_args = {
            'data': data_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': 16,
            'patience': 10,
            'save': True,
            'verbose': True,
            'project': 'id_card_detection',
            'name': 'yolo_id_detector_v2',
            
            # Data augmentation parameters
            'augment': True,           # Enable data augmentation
            'hsv_h': 0.015,           # Image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,             # Image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,             # Image HSV-Value augmentation (fraction)
            'degrees': 10.0,          # Image rotation (+/- deg)
            'translate': 0.1,         # Image translation (+/- fraction)
            'scale': 0.5,             # Image scale (+/- gain)
            'shear': 0.0,             # Image shear (+/- deg)
            'perspective': 0.0,       # Image perspective (+/- fraction), range 0-0.001
            'flipud': 0.0,            # Image flip up-down (probability)
            'fliplr': 0.5,            # Image flip left-right (probability)
            'mosaic': 1.0,            # Image mosaic (probability)
            'mixup': 0.0,             # Image mixup (probability)
            'copy_paste': 0.0,        # Segment copy-paste (probability)
            
            # Optimization parameters
            'lr0': 0.01,              # Initial learning rate
            'lrf': 0.01,              # Final learning rate (lr0 * lrf)
            'momentum': 0.937,        # SGD momentum/Adam beta1
            'weight_decay': 0.0005,   # Optimizer weight decay
            'warmup_epochs': 3.0,     # Warmup epochs
            'warmup_momentum': 0.8,   # Warmup initial momentum
            'warmup_bias_lr': 0.1,    # Warmup initial bias lr
            'box': 7.5,               # Box loss gain
            'cls': 0.5,               # CLS loss gain
            'dfl': 1.5,               # DFL loss gain
        }
        
        # Update with any additional arguments passed
        train_args.update(kwargs)
        
        print("🔧 Training with data augmentation enabled")
        print(f"📊 Number of classes: 2 (ID, Picture)")
        
        # Train the model
        results = self.model.train(**train_args)
        self.training_results = results
        
        print("✅ Training completed!")
        
        # Generate training plots
        self.plot_training_metrics()
        
        return results
    
    def plot_training_metrics(self):
        """Generate comprehensive training metrics plots"""
        if self.training_results is None:
            print("❌ No training results available to plot")
            return
        
        try:
            # Get the results directory
            results_dir = Path(self.training_results.save_dir)
            
            # Load the results CSV file
            results_file = results_dir / 'results.csv'
            if results_file.exists():
                results_df = pd.read_csv(results_file)
                
                # Create comprehensive plots
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle('YOLO Training Metrics - ID Card Detection', fontsize=16, fontweight='bold')
                
                # Plot 1: Loss curves
                if 'train/box_loss' in results_df.columns:
                    axes[0, 0].plot(results_df['epoch'], results_df['train/box_loss'], label='Train Box Loss', linewidth=2)
                    axes[0, 0].plot(results_df['epoch'], results_df['val/box_loss'], label='Val Box Loss', linewidth=2)
                    axes[0, 0].set_title('Box Loss')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                
                if 'train/cls_loss' in results_df.columns:
                    axes[0, 1].plot(results_df['epoch'], results_df['train/cls_loss'], label='Train Class Loss', linewidth=2)
                    axes[0, 1].plot(results_df['epoch'], results_df['val/cls_loss'], label='Val Class Loss', linewidth=2)
                    axes[0, 1].set_title('Classification Loss')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Loss')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                
                if 'train/dfl_loss' in results_df.columns:
                    axes[0, 2].plot(results_df['epoch'], results_df['train/dfl_loss'], label='Train DFL Loss', linewidth=2)
                    axes[0, 2].plot(results_df['epoch'], results_df['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
                    axes[0, 2].set_title('DFL Loss')
                    axes[0, 2].set_xlabel('Epoch')
                    axes[0, 2].set_ylabel('Loss')
                    axes[0, 2].legend()
                    axes[0, 2].grid(True, alpha=0.3)
                
                # Plot 2: Precision and Recall
                if 'metrics/precision(B)' in results_df.columns:
                    axes[1, 0].plot(results_df['epoch'], results_df['metrics/precision(B)'], 
                                   label='Precision', linewidth=2, color='green')
                    axes[1, 0].set_title('Precision')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Precision')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].set_ylim(0, 1)
                
                if 'metrics/recall(B)' in results_df.columns:
                    axes[1, 1].plot(results_df['epoch'], results_df['metrics/recall(B)'], 
                                   label='Recall', linewidth=2, color='orange')
                    axes[1, 1].set_title('Recall')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Recall')
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].set_ylim(0, 1)
                
                # Plot 3: mAP metrics
                if 'metrics/mAP50(B)' in results_df.columns:
                    axes[1, 2].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], 
                                   label='mAP@0.5', linewidth=2, color='red')
                    axes[1, 2].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], 
                                   label='mAP@0.5:0.95', linewidth=2, color='purple')
                    axes[1, 2].set_title('mAP Metrics')
                    axes[1, 2].set_xlabel('Epoch')
                    axes[1, 2].set_ylabel('mAP')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)
                    axes[1, 2].set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(results_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✅ Training metrics plots saved to: {results_dir / 'training_metrics.png'}")
                
                # Print final metrics
                self._print_final_metrics(results_df)
                
            else:
                print("❌ Results CSV file not found")
                
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def _print_final_metrics(self, results_df):
        """Print final training metrics"""
        if len(results_df) == 0:
            return
        
        final_epoch = results_df.iloc[-1]
        
        print("\n📊 FINAL TRAINING METRICS:")
        print("=" * 50)
        
        # Losses
        if 'train/box_loss' in final_epoch:
            print(f"📦 Final Box Loss: {final_epoch['train/box_loss']:.4f} (train) / {final_epoch['val/box_loss']:.4f} (val)")
        if 'train/cls_loss' in final_epoch:
            print(f"🎯 Final Class Loss: {final_epoch['train/cls_loss']:.4f} (train) / {final_epoch['val/cls_loss']:.4f} (val)")
        
        # Metrics
        if 'metrics/precision(B)' in final_epoch:
            print(f"🎯 Final Precision: {final_epoch['metrics/precision(B)']:.4f}")
        if 'metrics/recall(B)' in final_epoch:
            print(f"🔍 Final Recall: {final_epoch['metrics/recall(B)']:.4f}")
        if 'metrics/mAP50(B)' in final_epoch:
            print(f"📈 Final mAP@0.5: {final_epoch['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in final_epoch:
            print(f"📊 Final mAP@0.5:0.95: {final_epoch['metrics/mAP50-95(B)']:.4f}")
        
        print("=" * 50)
    
    def detect(self, image, confidence=0.5):
        """
        Detect ID cards and Pictures in an image
        
        Args:
            image: numpy array or image path
            confidence: detection confidence threshold
        
        Returns:
            YOLO results object with detections
        """
        results = self.model(image, conf=confidence)
        return results

    def detect_id_cards(self, image_path, confidence=0.5):
        """
        Detect ID cards and Pictures with structured results
        
        Args:
            image_path: path to image file
            confidence: detection confidence threshold
        
        Returns:
            List of detected objects with bounding boxes and class information
        """
        results = self.detect(image_path, confidence)
        
        detections = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    class_id = int(box.cls.item())
                    class_name = 'ID' if class_id == 0 else 'Picture'
                    
                    detections.append({
                        'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        'confidence': box.conf.item(),
                        'class_id': class_id,
                        'class': class_name
                    })
        
        return detections

    def detect_in_pdf(self, pdf_path, output_dir='extracted_cards', confidence=0.5):
        """
        Detect ID cards and Pictures in PDF document
        
        Args:
            pdf_path: path to PDF file
            output_dir: directory to save extracted cards
            confidence: detection confidence threshold
        
        Returns:
            List of detected objects
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_detections = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            
            # Convert to numpy array for OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detect objects on this page
            detections = self.detect_id_cards(img, confidence)
            for detection in detections:
                detection['page'] = page_num + 1
                all_detections.append(detection)
                
                # Extract and save the detected object
                x1, y1, x2, y2 = map(int, detection['bbox'])
                card_img = img[y1:y2, x1:x2]
                
                class_name = detection['class']
                cv2.imwrite(f"{output_dir}/{class_name}_page{page_num+1}_{len(all_detections)}.png", card_img)
        
        doc.close()
        return all_detections