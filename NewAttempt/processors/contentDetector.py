import cv2
import numpy as np

class ContentDetector:
    def __init__(self, id_processor, table_processor, image_processor):
        self.id_processor = id_processor
        self.table_processor = table_processor
        self.image_processor = image_processor
    
    def detect_content_type(self, image):
        "Detect if content is text, table, or image"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # First check for ID cards or passports - PASS image_processor for photo detection
        document_type = self.id_processor._detect_specific_documents(image, self.image_processor)
        if document_type:
            return document_type
        
        # Check for table structure 
        if self.table_processor._is_table(gray):
            return 'table'
        
        # Check for image
        if self.image_processor._is_image_content(gray):
            return 'image'
        
        # Default to text if no other type is detected
        return 'text'
    
    def classify_image_content(self, image_region):
        "Classify image content type"
        return self.image_processor.classify_image_content(image_region)