# config.py
import os

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/Output folders
INPUT_FOLDER = '/Users/emmafarigoule/Desktop/CICR/documents&data/data&brainstorm/data/ID testing'
OUTPUT_FOLDER = '/Users/emmafarigoule/Desktop/CICR/documents&data/extracted'

# Model paths
YOLO_MODEL_PATHS = ['/Users/emmafarigoule/Desktop/CICR/CICR---SPOC/best.pt']

# Training dataset paths (used by classifier/training_model.py)
TRAINING_IMAGES_DIR = '/Users/emmafarigoule/Desktop/CICR/id_card_detector/my_images'
TRAINING_LABELS_DIR = '/Users/emmafarigoule/Desktop/CICR/id_card_detector/my_labels'
TRAINING_OUTPUT_DIR = '/Users/emmafarigoule/Desktop/CICR/id_card_detector/dataset'

# ===== PROCESSING SETTINGS =====
DPI = 300  # PDF rendering resolution
OCR_LANGUAGES = ['en', 'es']
USE_GPU = False # Since I work on M3 Mac, no available GPU for tesseract
MIN_CONFIDENCE = 0.3

# ===== TRAINING SETTINGS =====
TRAINING_EPOCHS = 50
TRAINING_IMAGE_SIZE = 640
TRAINING_BATCH_SIZE = 16