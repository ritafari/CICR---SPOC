import cv2
import numpy as np

class Visualizer:
    def __init__(self, panel_width=600):
        self.panel_width = panel_width
    
    def create(self, image, ocr_results, content_type, extracted_text):
        """Create visualization image with side panel"""
        # Make copy for drawing
        if len(image.shape) == 2:
            viz_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            viz_image = image.copy()
        
        # Draw OCR bounding boxes
        for (bbox, text, prob) in ocr_results:
            if prob > 0.3:
                tl = (int(bbox[0][0]), int(bbox[0][1]))
                br = (int(bbox[2][0]), int(bbox[2][1]))
                
                # Different colors for different content
                if content_type == 'id_card':
                    color = (255, 165, 0)  # Orange
                elif content_type == 'picture':
                    color = (0, 0, 255)    # Red
                else:
                    color = (0, 255, 0)    # Green
                
                cv2.rectangle(viz_image, tl, br, color, 2)
                cv2.putText(viz_image, f"{prob:.2f}", (tl[0], tl[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Create side panel
        panel = self._create_text_panel(viz_image.shape[0], content_type, extracted_text)
        
        # Combine
        combined = np.hstack((viz_image, panel))
        return combined
    
    def _create_text_panel(self, height, content_type, text):
        """Create side panel with text"""
        panel = np.ones((height, self.panel_width, 3), dtype=np.uint8) * 240  # Light gray
        
        # Add header
        cv2.putText(panel, f"CONTENT: {content_type.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add extracted text (truncated)
        cv2.putText(panel, "EXTRACTED TEXT:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add text content
        y = 120
        lines = text.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            if y < height - 30:
                cv2.putText(panel, line[:50], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y += 25
        
        return panel