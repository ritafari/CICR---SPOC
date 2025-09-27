from extractors.BaseExtractor import BaseExtractor

class TextExtractor(BaseExtractor):
    """Extracts content from plain text files (.txt, .log, .md, etc.)."""
    
    def extract(self, file_path):
        """
        Reads and returns the content of a text file.
        Args:
            file_path (str): The path to the text file.
            
        Returns:
            str: The content of the file.
        """
        # 'utf-8' is a common encoding, but some files might have others.
        # errors='ignore' will prevent crashes if an unreadable character is found.
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"[Error] Failed to read text file {file_path}. Reason: {e}"
