import easyocr
import cv2
import numpy as np
import re 

class idProcessor:
    def __init__(self, reader):
        self.reader = reader
        # ID Card patterns and keywords (en, es, ar)    
        self.id_keywords = {
            'en': [
                'id card', 'identity card', 'driver license', 'driving licence',
                'passport', 'national id', 'employee id', 'student id',
                'license no', 'id no', 'card no', 'document no', 'expiry date', 
                'issued date'
            ],
            'es': [
                'carnet identidad', 'documento identidad', 'dni', 'cédula',
                'pasaporte', 'carnet conducir', 'licencia conducir', 'fecha expedición', 
                'fecha caducidad','documento nacional', 'carnet estudiante', 
                'carnet empleado', 'INSTITUTO FEDERAL ELECTORAL'
            ],
            'ar': [
                'بطاقة الهوية', 'جواز السفر', 'رخصة القيادة',
                'تاريخ الميلاد', 'تاريخ الانتهاء', 'تاريخ الإصدار',
                'العنوان', 'التوقيع', 'الصورة', 'الهوية الوطنية',
                'رقم الهوية', 'رقم الوثيقة'
            ]
        }

        # Universal patterns 
        self.id_patterns = [
            r'[A-Z0-9]{6,20}',  # ID numbers (works in most languages)
            r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',  # Date formats
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # Isolated dates
            r'[A-Z][a-z]+ [A-Z][a-z]+',  # Western full names
            r'[\u0600-\u06FF\s]{3,}',  # Arabic text (3+ Arabic characters)
            r'[\u00C0-\u024F\s]{3,}',  # Extended Latin for European languages
        ]

        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en', 'es'])

    def _detect_specific_documents(self, image, image_processor):
        "Detect specific document types like ID cards or passports"
        # Get OCR text for document type detection
        ocr_text = self._quick_ocr_detection(image)
        ocr_text_lower = ocr_text.lower()

        # Check for ID card - PASS image_processor
        if self._is_id_card(image, ocr_text, ocr_text_lower, image_processor):
            return 'id_card'
        
        # Check for passport 
        if self._is_passport(ocr_text_lower):
            return 'passport'
        
        return None

    def _is_id_card(self, image, ocr_text, ocr_text_lower, image_processor):
        "Detect if the content is an ID card using multilingual approach"
        # Check for ID-specific keywords in all languages
        total_keyword_matches = 0
        for lang, keywords in self.id_keywords.items():
            for keyword in keywords:
                if keyword in ocr_text_lower:
                    total_keyword_matches += 1
                    print(f"    🔍 Found ID keyword '{keyword}' in language '{lang}'")
        
        # Check for ID-specific patterns
        pattern_matches = 0
        for pattern in self.id_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                pattern_matches += len(matches)
                print(f"    🔍 Found ID pattern match: '{pattern}'")
        
        # Check document structure - USE image_processor for photo detection
        h, w = image.shape[:2]
        aspect_ratio = max(w / h, h / w)
        has_photo_region = image_processor._detect_photo_region(image)  # Fixed: use image_processor
        has_id_structure = self._has_id_card_structure(image, ocr_text)

        # Strong indication of ID card (relaxed threshold for multilingual)
        if (total_keyword_matches >= 1 and pattern_matches >= 3) or \
            (has_photo_region and (total_keyword_matches >= 1 or pattern_matches >= 3)) or \
            (has_id_structure and pattern_matches >= 1):
            return True
            
        return False

    def _has_id_card_structure(self, image, ocr_text):
        "Check if doc has typical ID card Structure"
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Check if aspect ratio matches ID card proportions
        id_card_ratio = 1.4 <= aspect_ratio <= 1.8
        
        # Check for multiple text blocks (typical in IDs)
        text_blocks = len(re.findall(r'\b[A-Z0-9]{4,}\b', ocr_text))
        
        # Check for common ID fields using multilingual patterns
        field_patterns = [
            r'name|nombre|nombre completo|اسم',  # Name field
            r'id|dni|cedula|identificacion|هوية',  # ID field
            r'birth|nacimiento|fecha.*nac|تاريخ.*ميلاد',  # Birth date
            r'expir|caducidad|vencimiento|انتهاء',  # Expiry
        ]
        
        field_matches = sum(1 for pattern in field_patterns if re.search(pattern, ocr_text, re.IGNORECASE))
        
        return id_card_ratio and text_blocks >= 3 and field_matches >= 2
    
    def _is_passport(self, ocr_text_lower):
        "Detect passport documents in multiple languages"
        passport_keywords = [
            # English
            'passport', 'passeport', 'passaporto', 'pasaporte',
            # Spanish
            'pasaporte', 
            # Arabic (transliterated)
            'jawaz', 'safar', 'جواز', 'سفر',
            # Other common terms
            'pass', 'reisepass'
        ]
        
        return any(keyword in ocr_text_lower for keyword in passport_keywords)

    def _quick_ocr_detection(self, image):
        "Perform a quick OCR for document type detection"
        try:
            # Use a smaller image for faster processing
            small_image = cv2.resize(image, (800,800)) if max(image.shape) > 800 else image
            results = self.reader.readtext(small_image, detail=0)
            return ' '.join(results)
        except:
            return ""