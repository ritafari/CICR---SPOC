from extractors.BaseExtractor import BaseExtractor

# I choose PyMuPDF (fitz) because it's fast, lightweight, and has good text extraction capabilities AND data processing occurs LOCALLY (see restrictions for project)
# MAKE SURE TO INSTALL PyMuPDF: pip install PyMuPDF !!!!!!!
# PyMuPDF uses a package named fitz internally, so we import fitz from PyMuPDF

try:
    import fitz 
except ImportError:
    fitz = None

class PdfExtractor(BaseExtractor):
    """EXtracts etxt content PDF files using PyMuPDF (fitz) library"""
        
    def __init__(self, model_size=None):
        """Initialize the PdfExtractor
        Args:
            model_size: Not used here but kept for compatibility with other extractors
        """
        if not fitz:
            raise "[Warning] PyMuPDF library is not installed. Please install it with 'pip install PyMuPDF'"
        
    def extract(self, file_path):
        """Extracts text from a PDF file.
        Args:
            file_path (str): The path to the PDF file.
        Returns:
            str: The extracted text from the PDF file.
        """
        if not fitz:
            raise "[Warning] PyMuPDF library is not installed. Please install it with 'pip install PyMuPDF'"
        
        text_content = []

        try:
            # Open the PDF file
            pdf_document = fitz.open(file_path)
    
            # Iterate through each page and extract text
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                # page.get_text() extracts the text from the page
                text = page.get_text()
                if text:
                    text_content.append(text)

            # Important to close the document after processing
            pdf_document.close()
            
            
        except Exception as e:
            return f"[Error] Failed to process PDF file {file_path}. Reason: {e}"

        return "\n".join(text_content)
    
# Next step would be to integrate an OCR step for scanned PDFs, but that would require additional libraries and processing time.
# --> could we just import .image_extractor from ImageExtractor and use it here for each page converted to image?
# Also next next step would be to translate all PDFs to desired language
# For now, this extractor works well for text-based PDFs.
