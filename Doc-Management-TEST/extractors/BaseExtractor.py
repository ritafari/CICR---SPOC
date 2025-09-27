from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """Abstract Base Class (Interface) for all Extractors
    
    This class defines a contract that all other extractors must follow.
    They have to have an extract method that takes a file path as input and returns extracted text.
    This is so that the main program can use any extractor in the exact same way, without needing to know it's specific implementation details.
    """
    @abstractmethod
    def extract(self, file_path):
        """Extracts text from the given file path.
        
        Args:
            file_path (str): The path to the file to be processed.
        """
        pass