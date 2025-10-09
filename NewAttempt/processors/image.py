import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass
    
    def _detect_photo_region(self, image):
        "Detect photo region in ID cards using face detection"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for rectangular regions that could be photos
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000: # Minimum area for photo region
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.7 <= aspect_ratio <= 1.3 and area > 0.05 * (image.shape[0] * image.shape[1]):
                    return True
        return False

    def _is_image_content(self, gray_image):
        "Detect if content is primarily image (graphical/non-text)"
        # Use edge detection to find non-text areas
        edges = cv2.Canny(gray_image, 100, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # High density often indicates image content
        return edge_density > 0.1
    
    def classify_image_content(self, image_region):
        "Classify image content type with document awareness"
        # Note: You'll need to pass the content detector or handle document detection differently
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)

        # Edge detection for image content
        edges = cv2.Canny(gray, 100, 150)
        edge_density = np.sum(edges > 0) / edges.size

        if edge_density > 0.15:
            return 'Chart/Graph'
        elif edge_density > 0.08:
            return 'Diagram'
        else:
            return 'Photograph'