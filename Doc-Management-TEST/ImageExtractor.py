from .base_extractor import BaseExtractor

# Here use OCR (optical character recognition) to extract text from images
# Tesseract (runs entirely locally) and is a powerful OCR engine that supports multiple languages (including arabic and spanish accroding to their doc)
# 1. Install Tesseract OCR on your system
# 2. Install pytesseract and Pillow libraries: pip install pytesseract Pillow
# VOC:
# OSD = Orientation and Script Detection

try:
    import pytesseract
    from pytesseract import Output
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

class ImageExtractor(BaseExtractor):
    """Extracts text from an image file using OCR"""

    def __init__(self, model_size=None):
        """Initialize the ImageExtractor
        Args:
            model_size: Not used here but kept for compatibility with other extractors
        """
        if not pytesseract or not Image:
            raise "[Warning] pytesseract or Pillow library is not installed. Please install them with 'pip install pytesseract Pillow'"
    

    def detect_language(self, file_path):
        """Detect language in the image with Tesseract.
        Args:
            file_path (str): The path to the image file.
        Returns:
            str: The extracted text from the image file.
        """
        try:
            # Open the image file
            img = Image.open(file_path)

            # Try to recognize language with tesseract's built-in language detection
            data = pytesseract.image_to_osd(img)

            # Parse the OSD data to find the script/language
            for line in data.split('\n'):
                if 'Script:' in line:
                    script = line.split(':')[-1].strip()
                    # Map script to Tesseract language code
                    script_to_lang = {
                        'Latin': 'eng',  # English and many other languages
                        'Arabic': 'ara',
                        'Cyrillic': 'rus',
                        'Devanagari': 'hin',
                        'Han': 'chi_sim',  # Simplified Chinese
                        'Hiragana': 'jpn',
                        'Katakana': 'jpn',
                        'Hangul': 'kor',
                        'Greek': 'ell',
                        'Hebrew': 'heb',
                        'Spanish': 'spa',
                    }
                    return script_to_lang.get(script, 'eng')
            return 'eng'  # Default to English if script not found

        except Exception as e:
            lang = 'eng'  # Default to English if detection fails
            

    def extract(self, file_path, language = None, translate_to_english = False):
        """Extracts text from an image file using OCR.
        Args:
            file_path (str): The path to the image file.
            language (str, optional): The language of the text in the image. If None, the method will attempt to detect the language automatically.
            translate_to_english (bool, optional): If True, the extracted text will be translated to English. Defaults to False.
        Returns:
            str: The extracted text from the image file.
        """
        if not pytesseract or not Image:
            raise "[Warning] Required libraries are not installed."
        
        try:
            # Open the image file
            img = Image.open(file_path)

            # If language is not provided, attempt to detect it
            if language is None:
                lang = self.detect_language(file_path)
                print(f"[Info] Detected language: {language}")
            else:
                lang = language

            # Configure Tesseract
            config = '--oem 3 --psm 6'  # OEM 3 = Default, PSM 6 = Assume a single uniform block of text

            # Add translation
            if translate_to_english and lang != 'eng':
                config += ' -c translate=1'
                print (f"[Info] Translating to English enabled for language:{lang}.")
            
            # Perform OCR
            text = pytesseract.image_to_string(img, lang=lang, config=config)
            
            return text.strip() #instead of return text to remove any leading/trailing whitespace

        except Exception as e:
            return f"[Error] Failed to process image file {file_path}. Reason: {e}"
