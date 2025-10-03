import os

# Dictionary that maps file extensions to a standardized type name
EXTENSION_MAP = {
    # all text files
    '.txt': 'text',
    '.md': 'text',
    '.rtf': 'text',
    '.csv': 'text',
    '.log': 'text',
    '.json': 'text',
    '.xml': 'text',

    # all document files
    '.docx': 'docx',
    '.pdf': 'pdf',

    # all image files
    '.jpg': 'image',
    '.jpeg': 'image',
    '.png': 'image',
    '.gif': 'image',
    '.bmp': 'image',
    '.tiff': 'image',

    # all audio files (for speech-to-text)
    '.mp3': 'audio',
    '.wav': 'audio',
    '.flac': 'audio',
    '.m4a': 'audio',

    # all video files (Audio will first be extracted than speech-to-text)
    '.mp4': 'video',
    '.avi': 'video',
    '.mov': 'video',
}

def identify_file_type(file_path):
    """
    Identify the type of a file based on its extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The identified file type ('text', 'document', 'image', 'audio', 'video', or 'unknown').
    """
    # os.path.splitext splits "path/to/file.txt" into ("path/to/file", ".txt")  
    _, extension = os.path.splitext(file_path)
    
    # Return the type from the map, or 'unknown if the extension is not found
    # i used ext.lower() just to make it case insensitive (in case it's PDF instead of pdf)
    return EXTENSION_MAP.get(extension.lower(), 'unknown')