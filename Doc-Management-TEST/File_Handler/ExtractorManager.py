# This file is like the "manager"
# Has to know about all the different extractor tools and provde the correct one when needed

from extractors.PlainTextExtractor import TextExtractor
from extractors.AudioExtractor import AudioExtractor
from extractors.VideoExtractor import VideoExtractor
from extractors.PdfExtractor import PdfExtractor
from extractors.ImageExtractor import ImageExtractor
# If we create more extractors we just have to put them there and add them to the map below

# Dictionary that maps the standardized file type to the correct extrcator class
# Kinda like a central registry for our "tools"
EXTRACTOR_MAPPING = {
    'text': TextExtractor,
    'audio': AudioExtractor,
    'video': VideoExtractor,
    'pdf': PdfExtractor,
    'image': ImageExtractor,
}

def get_extractor(file_type):
    """
    Get the appropriate extrcator class based on the file type.
    Args:
        file_type (str): The standardized file type ('text', 'document', 'image', 'audio', 'video').
    Returns:
        An instance of an extractor class
    Raises:
        ValueError: If the file type is unknown or unsupported.
    """
    extractor_class = EXTRACTOR_MAPPING.get(file_type)
    
    if not extractor_class:
        raise ValueError(f"[Error] No extractor found for file type: {file_type}")
    
    # Return an instance of the extractor class
    return extractor_class()  # You can pass parameters here if needed, like model_size for AudioExtractor