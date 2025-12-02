import fitz  # PyMuPDF
import cv2
import numpy as np

class PDFLoader:
    def __init__(self, dpi=300):
        self.dpi = dpi
    
    def load(self, pdf_path):
        """Load PDF document"""
        try:
            return fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return None
    
    def extract_page_image(self, doc, page_num):
        """Extract a page as OpenCV image"""
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=self.dpi)
            img_data = pix.tobytes("png")
            image_np = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ Error extracting page {page_num}: {e}")
            return None