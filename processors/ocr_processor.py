import easyocr
import cv2

class OCRProcessor:
    def __init__(self):
        print("Loading EasyOCR model...")
        self.reader = easyocr.Reader(['en', 'es'], gpu=False)
    
    def extract_text(self, image):
        """Extract text from image"""
        try:
            # Ensure image is grayscale for OCR
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            results = self.reader.readtext(image)
            return results
        except Exception as e:
            print(f"❌ OCR error: {e}")
            return []