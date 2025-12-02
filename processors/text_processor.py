class TextPostprocessor:
    def process(self, ocr_results, content_type):
        """Process OCR results based on content type"""
        if not ocr_results:
            return "No text detected"
        
        # Extract all text
        all_text = ' '.join([text for _, text, confidence in ocr_results if confidence > 0.3])
        
        # Add content type header
        if content_type == 'id_card':
            return f"[ID CARD]\n{all_text}"
        elif content_type == 'picture':
            return f"[PICTURE]\n{all_text}"
        else:
            return all_text